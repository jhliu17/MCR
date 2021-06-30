import abc
import dill
import typing
from pathlib import Path
import matchzoo as mz
from matchzoo.engine.base_preprocessor import BasePreprocessor


class BasePipeline(metaclass=abc.ABCMeta):
    """
    :class:`BasePipeline` to input handle data.

    A Pipeline is a collection including multiple preprocessors

    """

    DATA_FILENAME = 'pipeline.dill'

    @abc.abstractmethod
    def fit(
        self,
        data_pack: 'mz.DataPack',
        verbose: int = 1
    ) -> 'BasePipeline':
        """
        Fit parameters on input data.

        This method is an abstract base method, need to be
        implemented in the child class.

        This method is expected to return itself as a callable
        object.

        :param data_pack: :class:`Datapack` object to be fitted.
        :param verbose: Verbosity.
        """

    @abc.abstractmethod
    def transform(
        self,
        data_pack: 'mz.DataPack',
        verbose: int = 1
    ) -> 'mz.DataPack':
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be
        implemented in the child class.

        :param data_pack: :class:`DataPack` object to be transformed.
        :param verbose: Verbosity.
            or list of text-left, text-right tuples.
        """

    def fit_transform(
        self,
        data_pack: 'mz.DataPack',
        verbose: int = 1
    ) -> 'mz.DataPack':
        """
        Call fit-transform.

        :param data_pack: :class:`DataPack` object to be processed.
        :param verbose: Verbosity.
        """
        return self.fit(data_pack, verbose=verbose).transform(data_pack, verbose=verbose)

    def state_dict(self):
        save_context = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BasePreprocessor):
                save_context[k] = v.state_dict()
        return save_context

    def save(self, dirpath: typing.Union[str, Path]):
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(self.DATA_FILENAME)

        if not dirpath.exists():
            dirpath.mkdir(parents=True)

        dill.dump(self.state_dict(), open(data_file_path, mode='wb'))

    def load_state_dict(self, save_context):
        for k, v in save_context.items():
            preprocessor = getattr(self, k)
            preprocessor.load_state_dict(v)

    def load(self, dirpath: typing.Union[str, Path]):
        """
        Load the fitted `context`. The reverse function of :meth:`save`.

        :param dirpath: directory path of the saved model.
        :return: a :class:`DSSMPreprocessor` instance.
        """
        dirpath = Path(dirpath)

        data_file_path = dirpath.joinpath(self.DATA_FILENAME)
        save_context = dill.load(open(data_file_path, 'rb'))
        self.load_state_dict(save_context)

    def info(self):
        s = []
        for k, v in self.__dict__.items():
            if isinstance(v, BasePreprocessor):
                s.append('%s vocab size: %d' % (k, v.vocab_size))
        return ', '.join(s)
