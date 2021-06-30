"""Basic Preprocessor."""
import matchzoo as mz

from matchzoo import DataPack
from matchzoo.engine.base_pipeline import BasePipeline
from matchzoo.helper import logger
from matchzoo.preprocessors.text_field_preprocessor import TextFieldPreprocessor


class LazadaFieldPreprocessor(TextFieldPreprocessor):
    def __init__(self, lang, *args, **kwargs):
        self.lang = lang
        super().__init__(*args, **kwargs)

    def _default_units(self) -> list:
        """Prepare needed process units."""
        return [
            mz.preprocessors.units.tokenize.ICUTokenize(lang=self.lang),
            mz.preprocessors.units.lowercase.Lowercase(),
        ]


class RHPPipeline(BasePipeline):
    def __init__(self,
                 language,
                 prd_filter_low_freq,
                 rvw_filter_low_freq):
        """Initialization."""
        logger.info(f"Data language in {language}")
        if language in ('en', 'english'):
            self.prd_text_field = TextFieldPreprocessor(
                field='text_left',
                mode='left',
                filter_mode='tf',
                filter_low_freq=prd_filter_low_freq
            )
            self.rvw_text_field = TextFieldPreprocessor(
                field='text_right',
                mode='right',
                filter_mode='tf',
                filter_low_freq=rvw_filter_low_freq
            )
        elif language in ('id', 'indonesian'):
            self.prd_text_field = LazadaFieldPreprocessor(
                lang=language,
                field='text_left',
                mode='left',
                filter_mode='tf',
                filter_low_freq=prd_filter_low_freq
            )
            self.rvw_text_field = LazadaFieldPreprocessor(
                lang=language,
                field='text_right',
                mode='right',
                filter_mode='tf',
                filter_low_freq=rvw_filter_low_freq
            )
        else:
            raise ValueError(f"Language {language} is invalid ...")

    def fit(self, data_pack: DataPack, verbose: int = 1):
        self.prd_text_field.fit(data_pack, verbose=verbose)
        self.rvw_text_field.fit(data_pack, verbose=verbose)
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create truncated length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = self.prd_text_field.transform(data_pack, verbose=verbose)
        data_pack = self.rvw_text_field.transform(data_pack, verbose=verbose)
        return data_pack
