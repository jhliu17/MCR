"""Text Field Preprocessor."""

import typing

import matchzoo as mz
from matchzoo import DataPack
from matchzoo.helper import logger
from matchzoo.engine.base_preprocessor import BasePreprocessor
from tqdm import tqdm

from . import units
from .build_unit_from_data_pack import build_unit_from_data_pack
from .build_vocab_unit import build_vocab_unit
from .chain_transform import chain_transform
from .units.vocabulary import Vocabulary

tqdm.pandas()


class TextFieldPreprocessor(BasePreprocessor):
    """
    Text field preprocessor helper. All the field in this proprocessor
    will share the state (eg. Vocabulary, Frequency Counter).

    :param field: String, indicates the processed filed of this preprocessor.
    :param mode: String, indicates where is the processed field (left or right).
    :param truncated_mode: String, mode used by :class:`TruncatedLength`.
        Can be 'pre' or 'post'.
    :param truncated_length_left: Integer, maximize length of :attr:`left`
        in the data_pack.
    :param filter_mode: String, mode used by :class:`FrequenceFilterUnit`. Can
        be 'df', 'cf', and 'idf'.
    :param filter_low_freq: Float, lower bound value used by
        :class:`FrequenceFilterUnit`.
    :param filter_high_freq: Float, upper bound value used by
        :class:`FrequenceFilterUnit`.
    """

    def __init__(self,
                 field: typing.Union[str, typing.List[str]],
                 mode: typing.Union[str, typing.List[str]],
                 truncated_mode: str = 'pre',
                 truncated_length: int = None,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 1,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False):
        """Initialization."""
        super().__init__()
        field = field if isinstance(field, list) else [field]
        mode = mode if isinstance(mode, list) else [mode]
        self._field_w_mode = list(zip(field, mode))
        self._truncated_mode = truncated_mode
        self._truncated_length = truncated_length
        self._filter_low_freq = filter_low_freq
        self._filter_high_freq = filter_high_freq
        self._filter_mode = filter_mode
        self._remove_stop_words = remove_stop_words

        # build the process units
        self._build_unit()
    
    def _build_unit(self):
        if self._truncated_length:
            self._truncatedlength_unit = units.TruncatedLength(
                self._truncated_length, self._truncated_mode
            )

        self._filter_unit = units.FrequencyFilter(
            low=self._filter_low_freq,
            high=self._filter_high_freq,
            mode=self._filter_mode
        )
        self._units = self._default_units()
        if self._remove_stop_words:
            self._units.append(units.stop_removal.StopRemoval())

    @classmethod
    def _default_units(cls) -> list:
        """Prepare needed process units."""
        return [
            mz.preprocessors.units.tokenize.Tokenize(),
            mz.preprocessors.units.lowercase.Lowercase()
        ]

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`BasicPreprocessor` instance.
        """
        # process each field
        for f, m in self._field_w_mode:
            data_pack = data_pack.apply_on_field(chain_transform(self._units), field=f,
                                                 mode=m, verbose=verbose)

        # jointly build filter
        fitted_filter_unit = build_unit_from_data_pack(self._filter_unit,
                                                       data_pack,
                                                       field_w_mode=self._field_w_mode,
                                                       flatten=False,
                                                       verbose=verbose)
        self._context['filter_unit'] = fitted_filter_unit

        # filter each field
        for f, m in self._field_w_mode:
            data_pack = data_pack.apply_on_field(fitted_filter_unit.transform, field=f,
                                                 mode=m, verbose=verbose)

        # jointly build vocab
        vocab_unit = build_vocab_unit(
            data_pack, field_w_mode=self._field_w_mode, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit
        vocab_size = len(vocab_unit.state['term_index'])

        # fit logger
        logger.info("Text field %s processed results:" % self.process_field)
        logger.info('Vocab size: %d' % vocab_size)

        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create truncated length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()

        for f, m in self._field_w_mode:
            data_pack.apply_on_field(chain_transform(self._units), field=f, mode=m,
                                     inplace=True, verbose=verbose)
            data_pack.apply_on_field(self._context['filter_unit'].transform, field=f,
                                     mode=m, inplace=True, verbose=verbose)
            data_pack.apply_on_field(self._context['vocab_unit'].transform, field=f,
                                     mode=m, inplace=True, verbose=verbose)

            if self._truncated_length:
                data_pack.apply_on_field(self._truncatedlength_unit.transform, field=f,
                                         mode=m, inplace=True, verbose=verbose)

            data_pack.append_field_length(
                field=f, mode=m, inplace=True, verbose=verbose)
            data_pack.drop_field_empty(field=f, mode=m, inplace=True)

        return data_pack

    @property
    def field(self):
        return (f for f, _ in self._field_w_mode)

    @property
    def mode(self):
        return (m for _, m in self._field_w_mode)

    def _get_vocab(self):
        for _, v in self.context.items():
            if isinstance(v, Vocabulary):
                return v
        return None

    @property
    def vocab(self):
        return self._get_vocab()

    @property
    def vocab_size(self):
        return len(self.vocab) if self.vocab else None

    @property
    def embedding_input_dim(self):
        return self.vocab_size

    @property
    def process_field(self):
        return ','.join('%s:%s' % (f, m) for f, m in self._field_w_mode)

    def state_dict(self):
        state = {}
        state['context_save'] = super().state_dict()
        state['_field_w_mode'] = self._field_w_mode
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state['context_save'])
        self._field_w_mode = state['_field_w_mode']
