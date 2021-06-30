import os
import torch
import matchzoo.losses
import matchzoo.datasets
import torch.distributed as dist

from matchzoo.data_pack import DataPack
from matchzoo.dataloader import InstanceDataset, InstanceDataLoader
from matchzoo.tasks import Ranking, Classification
from matchzoo.engine.base_modeling import BaseModeling
from matchzoo.trainers import ModelingTrainer, DistributedModelingTrainer
from matchzoo.helper import logger
from torch.utils.data import DistributedSampler


class BasicModeling(BaseModeling):
    '''
    A basic implementation of `BaseModeling`.

    * For the dataset, you must define the training set when the stage is in `train`.
    For the dev and test set, if you don't want to use it. Try to return `None`.

    * Support Distributed training and Data parallel setting.

    * Implement the most of method, while the remains are
    some methods closely couple with the specefic modeling problem.

    They are:
        def build_model(self) -> nn.Module:
        raise NotImplementedError

        def build_metrics(self) -> typing.List[BaseMetric]:
            """
            Set your metrics.
            """
            raise NotImplementedError

        def build_preprocessor(self) -> BasePreprocessor:
            raise NotImplementedError

        def load_data(self, dataset_class) -> typing.Tuple[DataPack]:
            """
            return (train_pack, dev_pack, test_pack)
            if no pack, return None
            """
            raise NotImplementedError

        def build_dataset_callback(self) -> typing.List[BaseCallback]:
            raise NotImplementedError

        def build_dataloader_callback(self) -> typing.List[BaseCallback]:
            raise NotImplementedError
    '''

    def build_task(self):
        task = self.config.train.task
        loss_argv = self.config.train.loss
        loss_class = getattr(matchzoo.losses, loss_argv['type'])
        logger.info('Run %s task with %s loss function' %
                    (task, loss_argv['type']))

        loss_argv.pop('type')
        if task == 'ranking':
            self.task = Ranking(losses=loss_class(**loss_argv.get_map()))
        elif task == 'classification':
            self.task = Classification(
                losses=loss_class(**loss_argv.get_map()))
        else:
            raise Exception(f'Task {task} did not be implemented')

        self.task.metrics = self.metrics

    def _load_processed_data(self, save_dir):
        logger.info(
            'Loading cached Datapack and Pipeline from %s...' % save_dir)
        # selective loading dataset
        train_pack_processed = DataPack.load(
            save_dir, 'train') if self.stage == 'train' else None
        dev_pack_processed = DataPack.load(
            save_dir, 'dev') if self.stage in ('train', 'dev') else None
        test_pack_processed = DataPack.load(
            save_dir, 'test') if self.stage == 'test' else None

        # load the preprocessor
        self.preprocessor.load(save_dir)
        return (train_pack_processed, dev_pack_processed, test_pack_processed)

    def _preprocess_data(self, save_dir):
        logger.info('Start processing %s dataset with cat %s...' % (self.config.data.dataset,
                                                                    self.config.data.cat))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # load raw data
        dataset_class = getattr(
            matchzoo.datasets, self.config.data.dataset)
        train_pack_raw, dev_pack_raw, test_pack_raw = self.load_data(
            dataset_class)

        # process and transform dataset
        train_pack_processed = self.preprocessor.fit_transform(
            train_pack_raw)
        dev_pack_processed = self.preprocessor.transform(dev_pack_raw)
        test_pack_processed = self.preprocessor.transform(test_pack_raw)

        # save dataset and preprocessor
        logger.info('Saving processed %s datapack and pipeline with cat %s to `%s`...' % (self.config.data.dataset,
                                                                                          self.config.data.cat, save_dir))
        train_pack_processed.save(save_dir, 'train')
        dev_pack_processed.save(save_dir, 'dev')
        test_pack_processed.save(save_dir, 'test')
        self.preprocessor.save(save_dir)
        return (train_pack_processed, dev_pack_processed, test_pack_processed)

    def build_data(self):
        save_dir = self.config.data.save_dir

        if os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0 \
                and not self.config.data.reprocess:
            """
            TODO: still a naive save dir checking method...
            """
            # selective loading dataset
            train_pack_processed, dev_pack_processed, \
                test_pack_processed = self._load_processed_data(save_dir)
        else:
            # only the main process need to pre-process the dataset
            if self.rank in (0, -1):
                train_pack_processed, dev_pack_processed, \
                    test_pack_processed = self._preprocess_data(save_dir)
                
                if self.rank == 0:
                    dist.barrier()
            else:
                # wait to load
                dist.barrier()
                train_pack_processed, dev_pack_processed, \
                    test_pack_processed = self._load_processed_data(save_dir)

        # show the training set desc
        if self.stage == 'train':
            head_num = (self.config.train.num_neg + 1) * 2
            logger.info('Processed datapack preview:\n')
            logger.info('Train left with %d samples:' % len(train_pack_processed.left))
            logger.info_format(train_pack_processed.left.head())
            logger.info('Train right with %d samples:' % len(train_pack_processed.right))
            logger.info_format(train_pack_processed.right.head())
            logger.info('Train relation with %d items, %d samples:' % (len(train_pack_processed.relation.id_left.unique()), len(train_pack_processed.relation)))
            logger.info_format(train_pack_processed.relation.head(head_num))
            logger.info('Train label:')
            logger.info_format(train_pack_processed.relation.label.unique())
            logger.info('Processed pipeline preview:')
            logger.info(self.preprocessor.info() + '\n')

        self.datapack['train'] = train_pack_processed
        self.datapack['dev'] = dev_pack_processed
        self.datapack['test'] = test_pack_processed

    def build_dataset(self):
        for k, v in self.datapack.items():
            if v is None:
                continue

            if k == 'train':
                logger.info('Build the %s dataset with %d batch size...' % (
                    k, self.config.train.batch_size * self.config.train.allocate_num
                )
                )
                dataset = InstanceDataset(
                    data_pack=self.datapack[k],
                    mode=self.config.train.mode,
                    num_dup=self.config.train.num_dup,
                    num_neg=self.config.train.num_neg,
                    shuffle=self.config.train.shuffle,
                    allocate_num=self.config.train.allocate_num,
                    resample=self.config.train.resample,
                    callbacks=self.dataset_callback.get(k, None),
                    weighted_sampling=self.config.train.weighted_sampling,
                    relation_building_interval=self.config.train.relation_building_interval,
                    relation_checkpoint=self.config.train.relation_checkpoint if self.config.train.contains('relation_checkpoint') else None
                )
            else:
                logger.info('Build the %s dataset with %d batch size...' % (
                    k, self.config.test.batch_size * self.config.test.allocate_num
                )
                )
                dataset = InstanceDataset(
                    data_pack=self.datapack[k],
                    shuffle=False,
                    allocate_num=self.config.test.allocate_num,
                    callbacks=self.dataset_callback.get(k, None)
                )
            self.dataset[k] = dataset

    def _build_distributed_sampler(self):
        """
        Sample split for distributed training.
        """
        for k, v in self.dataset.items():
            if v is None:
                continue
            if k == 'train':
                self.dataset_sampler[k] = DistributedSampler(v)

    def build_dataloader(self):
        train_batch_size = self.config.train.batch_size
        test_batch_size = self.config.test.batch_size
        worker = int(self.config.device_setting.num_workers)

        if self.rank != -1:
            self._build_distributed_sampler()

        for k, v in self.dataset.items():
            if v is None:
                continue
            logger.info('Build the %s dataloader with %d worker...' %
                        (k, worker))
            self.dataloader[k] = InstanceDataLoader(
                dataset=v,
                batch_size=train_batch_size if k == 'train' else test_batch_size,
                rank=self.rank,
                sampler=self.dataset_sampler[k],
                stage=k,
                num_workers=worker,
                callbacks=self.dataloader_callback
            )

    def build_optimizer(self):
        """
        Build the training optimizer.
        """
        if self.stage == 'train':
            optimizer_argv: dict = self.config.train.optimizer

            logger.info('Using %s optimizer' % optimizer_argv['type'])
            optimizer_class = getattr(torch.optim, optimizer_argv['type'])
            optimizer_argv.pop('type')
            self.optimizer = optimizer_class(
                params=self.model.parameters(),
                **optimizer_argv.get_map()
            )

    def build_scheduler(self):
        """
        Build the training scheduler.
        """
        if self.stage == 'train':
            scheduler_argv = self.config.train.scheduler
            if not scheduler_argv:
                return

            logger.info('Using %s scheduler' % scheduler_argv['type'])
            scheduler_class = getattr(
                torch.optim.lr_scheduler, scheduler_argv['type'])
            scheduler_argv.pop('type')
            self.scheduler = scheduler_class(
                optimizer=self.optimizer,
                **scheduler_argv.get_map()
            )

    def build_device(self):
        device_list = self.config.device_setting.visible_device_list

        if self.config.device_setting.device == 'cuda':
            if isinstance(device_list, list):
                if self.rank == -1:
                    # use DataParallel
                    self.device = device_list
                else:
                    # use DistributedDataParallel
                    word_size = dist.get_world_size()
                    n_gpu = len(device_list)
                    if word_size <= n_gpu:
                        self.device = device_list[self.rank]
                    else:
                        raise Exception(
                            "The GPU number %d is less than Process number %d" % (n_gpu, word_size))
            elif isinstance(device_list, str):
                if self.rank == -1:
                    # single card
                    self.device = int(device_list)
                else:
                    raise ValueError(
                        'Can not adopt single card to `DistributedDataParallel`')
            else:
                raise Exception(
                    f'Device {device_list} can not be recognized...')
        else:
            self.device = "cpu"

    def build_trainer(self):
        if self.rank == -1:
            trainer = ModelingTrainer(
                model=self.model,
                task=self.task,
                optimizer=self.optimizer,
                main_metric=self.config.train.main_metric,
                scheduler=self.scheduler,
                trainloader=self.dataloader['train'],
                validloader=self.dataloader['dev'],
                validate_interval=self.config.train.validate_interval,
                validate_at_epoch_end=self.config.train.validate_at_epoch_end,
                save_interval=self.config.train.save_interval,
                start_epoch=self.config.train.start_epoch,
                epochs=self.config.train.end_epoch,
                device=self.device,
                stage=self.stage,
                patience=self.config.train.early_stopping,
                save_dir=self.config.train.checkpoint.dir,
                save_all=True,
                checkpoint=self.ckpt,
                debug=self.config.log.debug
            )
        else:
            trainer = DistributedModelingTrainer(
                rank=self.rank,
                model=self.model,
                task=self.task,
                optimizer=self.optimizer,
                main_metric=self.config.train.main_metric,
                scheduler=self.scheduler,
                trainloader=self.dataloader['train'],
                validloader=self.dataloader['dev'],
                validate_interval=self.config.train.validate_interval,
                validate_at_epoch_end=self.config.train.validate_at_epoch_end,
                save_interval=self.config.train.save_interval,
                start_epoch=self.config.train.start_epoch,
                epochs=self.config.train.end_epoch,
                device=self.device,
                stage=self.stage,
                patience=self.config.train.early_stopping,
                save_dir=self.config.train.checkpoint.dir,
                save_all=True,
                checkpoint=self.ckpt
            )
        self.trainer = trainer

    def train(self):
        """
        Run trainer
        """
        self.trainer.run()

    def dev(self):
        """
        Only the main process run the dev set.
        """
        if self.dataloader['dev'] is not None:
            if self.rank in (-1, 0):
                self.info_metric(self.trainer.evaluate(self.dataloader['dev']))
        else:
            print('Dev set is None...')

    def test(self):
        """
        Only the main process run the test set.
        """
        if self.dataloader['test'] is not None:
            if self.rank in (-1, 0):
                self.info_metric(self.trainer.evaluate(
                    self.dataloader['test']))
        else:
            print('Test set is None...')

    def info_metric(self, result: dict):
        print('Evaluation metrics:\n%s' % ('\n'.join(
              f'{k}: {round(v, 4)}' for k, v in result.items())))
