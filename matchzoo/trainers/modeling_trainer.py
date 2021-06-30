import typing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from matchzoo import tasks
from matchzoo.helper import logger
from matchzoo.dataloader import DataLoader, InstanceDataLoader
from matchzoo.utils import AverageMeter, EarlyStopping
from matchzoo.trainers import Trainer
from matchzoo.engine.base_task import BaseTask
from matchzoo.utils import load_tensors_to, Timer


class ModelingTrainer(Trainer):
    """
    A new trainer decouple the model with the task.
    And the model is a pure nn.Module.

    :param model: A :class:`nn.Module` instance.
    :param task: A :class:`BaseTask` instance.
    :param optimizer: A :class:`optim.Optimizer` instance.
    :param trainloader: A :class`DataLoader` instance. The dataloader
        is used for training the model.
    :param validloader: A :class`DataLoader` instance. The dataloader
        is used for validating the model.
    :param device: The desired device of returned tensor. Default:
        if None, use the current device. If `torch.device` or int,
        use device specified by user. If list, use data parallel.
    :param start_epoch: Int. Number of starting epoch.
    :param epochs: The maximum number of epochs for training.
        Defaults to 10.
    :param validate_interval: Int. Interval of validation.
    :param scheduler: LR scheduler used to adjust the learning rate
        based on the number of epochs.
    :param clip_norm: Max norm of the gradients to be clipped.
    :param patience: Number fo events to wait if no improvement and
        then stop the training.
    :param key: Key of metric to be compared.
    :param checkpoint: A checkpoint from which to continue training.
        If None, training starts from scratch. Defaults to None.
        Should be a file-like object (has to implement read, readline,
        tell, and seek), or a string containing a file name.
    :param save_dir: Directory to save trainer.
    :param save_all: Bool. If True, save `Trainer` instance; If False,
        only save model. Defaults to False.
    :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = verbose, 2 = one log line per epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        task: BaseTask,
        optimizer: optim.Optimizer,
        trainloader: DataLoader,
        validloader: DataLoader,
        stage: str = 'train',
        device: typing.Union[torch.device, int, list, None] = None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval: typing.Optional[int] = None,
        validate_at_epoch_end: bool = False,
        save_interval: typing.Optional[int] = None,
        main_metric: int = None,
        scheduler: typing.Any = None,
        clip_norm: typing.Union[float, int] = None,
        patience: typing.Optional[int] = None,
        checkpoint: typing.Union[str, Path] = None,
        save_dir: typing.Union[str, Path] = None,
        save_all: bool = False,
        verbose: int = 1,
        debug: bool = False,
        **kwargs
    ):
        """Base Trainer constructor."""
        self._task = task
        self._stage = stage
        self._load_model(model, device)
        if stage == 'train':
            self._load_dataloader(
                trainloader, validloader, validate_interval
            )

        self._optimizer = optimizer
        self._main_metric = self._task.metrics[main_metric]
        self._scheduler = scheduler
        self._clip_norm = clip_norm
        self._criterions = self._task.losses

        self._early_stopping = EarlyStopping(
            patience=patience,
            key=self._main_metric
        ) if patience > 0 and main_metric else None
        if self._early_stopping is None:
            logger.info('Without using early stopping')
        else:
            logger.info('Using early stopping with patience %d' % patience)

        self._start_epoch = start_epoch
        self._epochs = epochs
        self._iteration = 0
        self._verbose = verbose
        self._save_all = save_all
        self._save_interval = save_interval
        self._debug = debug
        self._validate_at_epoch_end = validate_at_epoch_end

        self._load_path(checkpoint, save_dir)

    def _load_model(
        self,
        model: nn.Module,
        device: typing.Union[torch.device, int, list, None] = None
    ):
        """
        Load model.

        :param model: :class:`nn.Module` instance.
        :param device: The desired device of returned tensor. Default:
            if None, use the current device. If `torch.device` or int,
            use device specified by user. If list, use data parallel.
        """
        if not isinstance(model, nn.Module):
            raise ValueError(
                'model should be a `nn.Module` instance.'
                f' But got {type(model)}.'
            )

        self._model = model
        self._n_gpu = 0

        # setting device
        if isinstance(device, list) and len(device):
            self._n_gpu = len(device)
            self._device = torch.device("cuda")
        elif isinstance(device, int):
            torch.cuda.set_device(device)
            self._device = torch.device("cuda", device)
            self._n_gpu = 1
        else:
            self._device = torch.device("cpu")

        # move to device
        self._model.to(self._device)

        # multi-gpu training (should be after apex fp16 initialization)
        if self._n_gpu > 1:
            logger.info("Using DataParallel")
            self._model = torch.nn.DataParallel(self._model)

    def _load_path(
        self,
        checkpoint: typing.Union[str, Path],
        save_dir: typing.Union[str, Path],
    ):
        """
        Load save_dir and Restore from checkpoint.

        :param checkpoint: A checkpoint from which to continue training.
            If None, training starts from scratch. Defaults to None.
            Should be a file-like object (has to implement read, readline,
            tell, and seek), or a string containing a file name.
        :param save_dir: Directory to save trainer.

        """
        # if self._stage == 'train':
        if not save_dir:
            save_dir = Path('.').joinpath('save')
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        logger.info('Save the checkpoints to %s...' % save_dir)
        self._save_dir = Path(save_dir)

        # Restore from checkpoint
        if checkpoint:
            logger.info('Loading checkpoint from %s...' % checkpoint)
            if self._save_all:
                self.restore(checkpoint)
            else:
                self.restore_model(checkpoint)

    def _load_tensor(self, model_input, model_output=None):
        """
        Tensor scheduler
        """
        model_input = load_tensors_to(model_input, self._device)

        if model_output is not None:
            model_output = load_tensors_to(model_output, self._device)
            return model_input, model_output
        else:
            return model_input

    def _run_scheduler(self, metrics: dict):
        """Run scheduler."""
        if self._scheduler and self._main_metric:
            self._scheduler.step(metrics[self._main_metric])

    def _run_early_stopping(self, metrics: dict):
        if self._early_stopping:
            self._early_stopping.update(metrics)
            if self._early_stopping.should_stop_early:
                logger.info(
                    'Ran out of patience. Early stoping at epoch %d - %d' % (
                        self._epoch, self._iteration
                    )
                )
                self._save('end_of_training')
            elif self._early_stopping.is_best_so_far:
                logger.info(
                    'Have improvement, saving the best so far')
                file_name = str(self._main_metric).replace(
                    '(', '_').replace(')', '')
                self._save('best_%s' % file_name)
            else:
                logger.info('Accumulated early stopping patience is %d (max %d)' % (
                    self._early_stopping._epochs_with_no_improvement,
                    self._early_stopping._patience
                )
                )

    def _check_early_stopping(self):
        """
        Check whether reach the end of early stopping.
        """
        if self._early_stopping and self._early_stopping.should_stop_early:
            return True
        return False

    def run(self):
        """
        Train model.

        The processes:
            Run each epoch -> Run scheduler -> Should stop early?

        """
        timer = Timer()
        self._model.train()
        for epoch in range(self._start_epoch, self._epochs + 1):
            self._epoch = epoch
            self._run_epoch()
            if self._check_early_stopping():
                break
        if self._verbose:
            tqdm.write(f'Cost time: {timer.time}s')

    def _run_evaluate_end(self, metrics: dict):
        self._run_scheduler(metrics)
        self._run_early_stopping(metrics)

    def _run_epoch(self):
        """
        Run each epoch.

        The training steps:
            - Get batch and feed them into model
            - Get outputs. Caculate all losses and sum them up
            - Loss backwards and optimizer steps
            - Evaluation
            - Update and output result
            - Save

        """
        # Get total number of batch
        num_batch = len(self._trainloader)
        train_loss = AverageMeter()

        with tqdm(enumerate(self._trainloader), total=num_batch,
                  disable=not self._verbose) as pbar:
            for step, (inputs, target) in pbar:
                # load tensor to device
                inputs, target = self._load_tensor(inputs, target)
                outputs = self._model(inputs)
                
                # caculate all losses and sum them up
                loss = torch.sum(
                    *[c(outputs, target) for c in self._criterions]
                )
                self._backward(loss)
                train_loss.update(loss.item())

                # set progress bar
                pbar.set_description(
                    f'Epoch {self._epoch}/{self._epochs} Loss {train_loss.avg:.3f}')

                # update iteration
                self._iteration += 1

                # run validate
                if self._validate_interval and self._iteration % self._validate_interval == 0:
                    logger.info('Start evaluating at epoch %d - %d' %
                                (self._epoch, self._iteration))
                    self._run_validate()

                    # early stopping
                    if self._check_early_stopping:
                        break

                # run step lr scheduling
                self._run_step_scheduler()

                # debug mode
                if self._debug and step > 3:
                    break
        
        # Save model at the epoch end
        if not self._debug and self._save_interval and self._epoch % self._save_interval == 0:
            self._save('epoch_%d' % self._epoch)

        # Run validate at the epoch end
        logger.info('Finish training at epoch %d, Average loss: %.3f' %
                    (self._epoch, train_loss.avg))
        if not self._debug and self._validate_at_epoch_end:
            self._run_validate()
        logger.info('Finished epoch %d\n\n' % self._epoch)

    def _run_validate(self):
        result = self.evaluate(self._validloader, save_pred=False)
        logger.info('Evaluation metrics:\n%s\n' % ('\n'.join(
            f'{k}: {round(v, 4)}' for k, v in result.items())))
        self._run_evaluate_end(result)

    def _run_step_scheduler(self):
        pass

    def _save(self, name=None):
        """Save."""
        if self._save_all:
            self.save(name)
        else:
            self.save_model(name)

    def save_model(self, name=None):
        """Save the model."""
        checkpoint = self._save_dir.joinpath(
            'model.pt' if name is None else 'model_%s.pt' % name)
        if hasattr(self._model, 'module'):
            torch.save(self._model.module.state_dict(), checkpoint)
        else:
            torch.save(self._model.state_dict(), checkpoint)

    def save(self, name=None):
        """
        Save the trainer.

        `Trainer` parameters like epoch, best_so_far, model, optimizer
        and early_stopping will be savad to specific file path.

        :param path: Path to save trainer.

        """
        checkpoint = self._save_dir.joinpath(
            'trainer.pt' if name is None else 'trainer_%s.pt' % name)
        if hasattr(self._model, 'module'):
            model = self._model.module.state_dict()
        else:
            model = self._model.state_dict()
        state = {
            'epoch': self._epoch,
            'model': model,
            'optimizer': self._optimizer.state_dict(),
        }
        if self._early_stopping:
            state['early_stopping'] = self._early_stopping.state_dict()
        if self._scheduler:
            state['scheduler'] = self._scheduler.state_dict()
        torch.save(state, checkpoint)

    def restore_model(self, checkpoint: typing.Union[str, Path]):
        """
        Restore model.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location='cpu')
        if hasattr(self._model, 'module'):
            self._model.module.load_state_dict(state)
        else:
            self._model.load_state_dict(state)

    def restore(self, checkpoint: typing.Union[str, Path] = None):
        """
        Restore trainer.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location='cpu')
        if hasattr(self._model, 'module'):
            self._model.module.load_state_dict(state['model'])
        else:
            self._model.load_state_dict(state['model'])
        if self._optimizer:
            self._optimizer.load_state_dict(state['optimizer'])
        self._start_epoch = state['epoch'] + 1
        if self._early_stopping:
            self._early_stopping.load_state_dict(state['early_stopping'])
        if self._scheduler:
            self._scheduler.load_state_dict(state['scheduler'])

    def predict(
        self,
        dataloader: DataLoader
    ) -> np.array:
        """
        Generate output predictions for the input samples.

        :param dataloader: input DataLoader
        :return: predictions

        """
        left = dataloader._dataset._data_pack.relation.id_left
        relation = dataloader._dataset._data_pack.relation
        logger.info("Evaluate data preview:")
        logger.info("Left num: %d" % len(left.unique()))
        logger.info("Relation num: %d" % len(relation))
        logger.info_format(relation.head(15))

        with torch.no_grad():
            self._model.eval()
            predictions = []
            for inputs, _ in tqdm(dataloader):
                inputs = self._load_tensor(inputs)
                outputs = self._model(inputs).detach().cpu()
                predictions.append(outputs)
            self._model.train()
            return torch.cat(predictions, dim=0).numpy()

    def evaluate(
        self,
        dataloader: InstanceDataLoader,
        save_pred: bool = True
    ):
        """
        Evaluate the model.

        :param dataloader: A DataLoader object to iterate over the data.

        """
        logger.info('Start evaluating...')
        result = dict()
        y_pred = self.predict(dataloader)
        y_true = dataloader.label
        id_left = dataloader.id_left

        if save_pred:
            id_right = dataloader.id_right
            self.save_pred(id_left, id_right, y_true, y_pred.reshape(-1).tolist())

        logger.info('Start calculating metrics...')
        if isinstance(self._task, tasks.Classification):
            for metric in self._task.metrics:
                result[metric] = metric(y_true, y_pred)
        else:
            for metric in self._task.metrics:
                result[metric] = self._eval_metric_on_data_frame(
                    metric, id_left, y_true, y_pred.squeeze(axis=-1))
        return result

    def save_pred(self, id_left: list, id_right: list, y_true: list, y_pred: list):
        logger.info('Saving the predict file...')
        data = {
            "id_left": id_left,
            "id_right": id_right,
            "y_true": y_true,
            "y_pred": y_pred
        }
        pred_frame = pd.DataFrame(data)
        # groups = pred_frame.sort_values(
        #             'y_pred', ascending=False).groupby('id_left')
        # id_group = []
        # for _, group in groups:
        #     id_group.append(group)
        # new_relation = pd.concat(id_group, ignore_index=True)
        pred_frame.to_json(self._save_dir.joinpath(f'{self._stage}.pred'))
