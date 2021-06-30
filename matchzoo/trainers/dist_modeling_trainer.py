import typing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from pathlib import Path
from matchzoo.dataloader import DataLoader
from matchzoo.trainers import ModelingTrainer
from matchzoo.engine.base_task import BaseTask
from matchzoo.helper import logger


class DistributedModelingTrainer(ModelingTrainer):
    """
    A Distributed compactable trainer. This trainer without using
    the metrics early stopping method and the scheduler must be
    a certain step scheduler. Otherwise, the scheduler would not be
    run correctly in distributed training.

    * support amp f16 training

    * This class supports `nn.DataParallel` training. However, the scheduler
    only supports step-wise scheduler. And the early stopping is not supported.

    :param rank: A int instance.
    :param fp16: A bool instance.
    """

    def __init__(
        self,
        rank: int,
        model: nn.Module,
        task: BaseTask,
        optimizer: optim.Optimizer,
        trainloader: DataLoader,
        validloader: DataLoader,
        stage: str = 'train',
        fp16: bool = False,
        fp16_opt_level: str = 'O1',
        device: typing.Union[torch.device, int, list, None] = None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval: typing.Optional[int] = None,
        save_interval: typing.Optional[int] = None,
        main_metric: int = None,
        scheduler: typing.Any = None,
        clip_norm: typing.Union[float, int] = None,
        patience: typing.Optional[int] = None,
        checkpoint: typing.Union[str, Path] = None,
        save_dir: typing.Union[str, Path] = None,
        save_all: bool = False,
        verbose: int = 1,
        **kwargs
    ):
        """Distributed Trainer constructor."""
        self._rank = rank
        self._fp16 = fp16
        self._fp16_opt_level = fp16_opt_level

        super().__init__(
            model=model,
            task=task,
            optimizer=optimizer,
            trainloader=trainloader,
            validloader=validloader,
            stage=stage,
            device=device,
            start_epoch=start_epoch,
            epochs=epochs,
            validate_interval=validate_interval,
            save_interval=save_interval,
            main_metric=main_metric,
            scheduler=scheduler,
            clip_norm=clip_norm,
            patience=-1,  # w/o eraly stopping
            checkpoint=checkpoint,
            save_dir=save_dir,
            save_all=save_all,
            verbose=verbose,
            **kwargs
        )

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

        # set fp16
        if self._fp16:
            try:
                from apex import amp
                # pass
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self._model, self._optimizer = amp.initialize(
                self._model, self._optimizer, opt_level=self._fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self._n_gpu > 1:
            logger.info("Using DataParallel")
            self._model = torch.nn.DataParallel(self._model)

        # Distributed training (should be after apex fp16 initialization)
        if self._rank != -1:
            logger.info("Using DistributedDataParallel")
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model,
                device_ids=[device],
                output_device=device,
                find_unused_parameters=True
            )

    def _run_validate(self):
        """
        Without considering early stopping and metrics-style
        scheduler.
        """
        # only the main process need to run evaluation
        if self._rank in (-1, 0):
            result = self.evaluate(self._validloader)
            logger.info('Evaluation metrics:\n%s\n' % ('\n'.join(
                f'{k}: {round(v, 4)}' for k, v in result.items())))

    def _run_step_scheduler(self):
        """
        Update step-wise lr scheduler.
        """
        self._scheduler.step(self._iteration)

    def _backward(self, loss):
        """
        Computes the gradient of current `loss` graph leaves.

        :param loss: Tensor. Loss of model.

        """
        self._optimizer.zero_grad()

        if self._n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

        if self._fp16:
            with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
            if self._clip_norm:
                nn.utils.clip_grad_norm_(
                    amp.master_params(self._optimizer), self._clip_norm
                )
        else:
            loss.backward()
            if self._clip_norm:
                nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._clip_norm
                )

        self._optimizer.step()

    def _save(self, name=None):
        """Save."""
        if self._rank in (-1, 0):
            if self._save_all:
                self.save(name)
            else:
                self.save_model(name)
