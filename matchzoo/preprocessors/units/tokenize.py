import nltk

from icu_tokenizer import Tokenizer as ICUTokenizer
from icu_tokenizer import Normalizer as ICUNormalizer
from .unit import Unit


class Tokenize(Unit):
    """Process unit for text tokenization."""

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        return nltk.word_tokenize(input_)


class ICUTokenize(Unit):
    def __init__(self, lang='en', norm_puncts=True):
        self._lang = lang
        self._normalizer = ICUNormalizer(lang=lang, norm_puncts=norm_puncts)
        self._tokenizer = ICUTokenizer(lang=lang)

    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        input_ = self._normalizer.normalize(input_)
        input_ = self._tokenizer.tokenize(input_)
        return input_