"""Basic data loader."""
import typing
import numpy as np
import torch
import torch.distributed as dist

from toolz.sandbox import unzip
from cytoolz import concat
from torch.utils import data
from matchzoo.dataloader.instance_dataset import InstanceDataset
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.dataloader import DataLoader


class InstanceDataLoader(DataLoader):
    """
    DataLoader that loads batches of data from a Dataset.

    :param dataset: The Dataset object to load data from.
    :param device: The desired device of returned tensor. Default: if None,
        use the current device. If `torch.device` or int, use device specified
        by user. If list, the first item will be used.
    :param stage: One of "train", "dev", and "test". (default: "train")
    :param callback: BaseCallback. See
        `matchzoo.engine.base_callback.BaseCallback` for more details.
    :param pin_momory: If set to `True`, tensors will be copied into
        pinned memory. (default: `False`)
    :param timeout: The timeout value for collecting a batch from workers. (
        default: 0)
    :param num_workers: The number of subprocesses to use for data loading. 0
        means that the data will be loaded in the main process. (default: 0)
    :param worker_init_fn: If not ``None``, this will be called on each
        worker subprocess with the worker id (an int in [0, num_workers - 1])
        as input, after seeding and before data loading. (default: None)

    Examples:
        >>> import matchzoo as mz
        >>> data_pack = mz.datasets.toy.load_data(stage='train')
        >>> preprocessor = mz.preprocessors.BasicPreprocessor()
        >>> data_processed = preprocessor.fit_transform(data_pack)
        >>> dataset = mz.dataloader.Dataset(
        ...     data_processed, mode='point', batch_size=32)
        >>> padding_callback = mz.dataloader.callbacks.BasicPadding()
        >>> dataloader = mz.dataloader.DataLoader(
        ...     dataset, stage='train', callback=padding_callback)
        >>> len(dataloader)
        4

    """

    def __init__(
        self,
        dataset: InstanceDataset,
        batch_size: int = 1,
        rank: int = -1,
        sampler=None,
        stage='train',
        callbacks: typing.List[BaseCallback] = None,
        pin_memory: bool = False,
        timeout: int = 0,
        num_workers: int = 0,
        worker_init_fn=None,
    ):
        """Init."""
        if stage not in ('train', 'dev', 'test'):
            raise ValueError(f"{stage} is not a valid stage type."
                             f"Must be one of `train`, `dev`, `test`.")
        if callbacks is None:
            callbacks = []

        self._dataset = dataset
        self._batch_size = batch_size
        self._sampler = sampler
        self._rank = rank
        self._sample_step = 0
        self._epoch = 0
        self._pin_momory = pin_memory
        self._timeout = timeout
        self._num_workers = num_workers
        self._worker_init_fn = worker_init_fn
        self._stage = stage
        self._callbacks = callbacks

        self.build_dataloader()

    def __len__(self) -> int:
        """Get the total number of batches."""
        return len(self._dataloader)

    def resample_dataset(self):
        """Resample the dataset."""
        # sample_step is to control the multiprocessing sampling pace.
        self._sample_step += 1

        if isinstance(self._dataset, data.DistributedSampler):
            self._dataset.dataset.resample_step(self._sample_step)
        else:
            self._dataset.resample_step(self._sample_step)

        # for distributed training
        if self._rank != -1:
            # make sure all processes have sync the dataset sampling
            dist.barrier()

    def build_dataloader(self):
        self._dataloader = data.DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
                sampler=self._sampler,
                num_workers=self._num_workers,
                pin_memory=self._pin_momory,
                timeout=self._timeout,
                worker_init_fn=self._worker_init_fn,
            )

    def __iter__(self) -> typing.Tuple[dict, torch.tensor]:
        """Iteration."""
        if self._stage == 'train' and self._epoch > 0:
            # resample before training, no need for the 0-th epoch,
            # because building the dataset has sampled a relation
            # table if needed.
            self.resample_dataset()

        for batch_data in self._dataloader:
            x, y = batch_data
            self._handle_callbacks_on_batch_unpacked(x, y)

            batch_x = {}
            for key, value in x.items():
                if 'id_left' in key or 'id_right' in key:
                    batch_x[key] = value
                    continue
                if 'left' not in key and 'right' not in key:
                    continue
                
                if isinstance(value, np.ndarray):
                    batch_x[key] = torch.from_numpy(value)
                else:
                    # print(key, x[key])
                    batch_x[key] = torch.tensor(value)

            if self._stage == 'test':
                yield batch_x, None
            else:
                if y.dtype == 'int':  # task='classification'
                    batch_y = torch.from_numpy(y).squeeze(dim=-1).long()
                else:  # task='ranking'
                    batch_y = torch.from_numpy(y).float()
                yield batch_x, batch_y

        self._epoch += 1

    def _collate_fn(self, instances):
        """
        instances: [(x, y), (x, y), ...]
        """
        x_zip, y_zip = unzip(instances)
        x_list = list(x_zip)
        y_list = list(y_zip)
        # gather_x = defaultdict(list)
        gather_x = {}
        gather_y = []
        dict_keys = x_list[0].keys()

        for key in dict_keys:
            gather_x[key] = list(concat(i[key] for i in x_list))

        gather_y = list(concat(i for i in y_list))
        gather_y = np.array(gather_y).reshape(-1, 1)

        return gather_x, gather_y

    def _handle_callbacks_on_batch_unpacked(self, x, y):
        for callback in self._callbacks:
            callback.on_batch_unpacked(x, y)

    @property
    def id_left(self) -> np.ndarray:
        """`id_left` getter."""
        # TODO: this patch is for deal with loading image callbacks
        # self._dataset._use_callbacks = False
        x, _ = self._dataset[:]
        # self._dataset._use_callbacks = True
        return x['id_left']

    @property
    def id_right(self) -> np.ndarray:
        """`id_left` getter."""
        # TODO: this patch is for deal with loading image callbacks
        # self._dataset._use_callbacks = False
        x, _ = self._dataset[:]
        # self._dataset._use_callbacks = True
        return x['id_right']

    @property
    def label(self) -> np.ndarray:
        """`label` getter."""
        # TODO: this patch is for deal with loading image callbacks
        # self._dataset._use_callbacks = False
        _, y = self._dataset[:]
        # self._dataset._use_callbacks = True
        return y.squeeze() if y is not None else None
