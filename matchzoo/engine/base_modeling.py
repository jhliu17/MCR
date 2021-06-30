import abc
import typing

from matchzoo.trainers import Trainer
from matchzoo.helper.configure import Configure
from matchzoo.engine.base_metric import BaseMetric
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.base_pipeline import BasePipeline
from matchzoo.engine.base_preprocessor import BasePreprocessor


class BaseModeling(abc.ABC):
    """
    Modeling is a base obejct to model the whole processing and training
    procedure with Matchzoo libarary.
    """

    def __init__(self, config: Configure, stage: str = 'train', ckpt: str = None, rank: int = -1):
        self.config = config
        if stage not in ('train', 'dev', 'test'):
            raise ValueError(
                f"{stage} is not a valid value ('train', 'dev', 'test') for modeling.")
        self.stage = stage
        self.ckpt = ckpt
        self.rank = rank

        self.metrics: typing.List[BaseMetric] = []
        self.task: typing.Optional[str] = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.trainer: typing.Optional[Trainer] = None
        self.preprocessor: typing.Union[BasePipeline, BasePreprocessor, None] = None
        self.device = None
        self.dataset_callback: typing.Optional[BaseCallback] = None
        self.dataloader_callback: typing.Optional[BaseCallback] = None
        self.datapack = {'train': None, 'dev': None, 'test': None}
        self.dataset = {'train': None, 'dev': None, 'test': None}
        self.dataset_sampler = {'train': None, 'dev': None, 'test': None}
        self.dataloader = {'train': None, 'dev': None, 'test': None}

        self.modeling()

    def modeling(self):
        '''
        Modeling sequence
        '''
        self.build_metrics()
        self.build_task()
        self.build_preprocessor()
        self.build_data()
        self.build_dataset_callback()
        self.build_dataset()
        self.build_dataloader_callback()
        self.build_dataloader()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_device()
        self.build_trainer()

    @abc.abstractmethod
    def build_metrics(self):
        ""

    @abc.abstractmethod
    def build_task(self):
        ""

    @abc.abstractmethod
    def build_data(self):
        ""

    @abc.abstractmethod
    def build_preprocessor(self):
        ""

    @abc.abstractmethod
    def build_dataset_callback(self):
        ""

    @abc.abstractmethod
    def build_dataset(self):
        ""

    @abc.abstractmethod
    def build_dataloader_callback(self):
        ""

    @abc.abstractmethod
    def build_dataloader(self):
        ""

    @abc.abstractmethod
    def build_model(self):
        ""

    @abc.abstractmethod
    def build_optimizer(self):
        ""

    @abc.abstractmethod
    def build_scheduler(self):
        ""

    @abc.abstractmethod
    def build_device(self):
        ""

    @abc.abstractmethod
    def build_trainer(self):
        ""

    @abc.abstractmethod
    def train(self):
        ""

    @abc.abstractmethod
    def dev(self):
        ""

    @abc.abstractmethod
    def test(self):
        ""
