"""Build unit from data pack."""
import typing
import matchzoo as mz

from tqdm import tqdm
from .units import StatefulUnit


def build_unit_from_data_pack(
    unit: StatefulUnit,
    data_pack: mz.DataPack, field_w_mode: typing.List[tuple] = None, mode: str = 'both',
    flatten: bool = True, verbose: int = 1
) -> StatefulUnit:
    """
    Build a :class:`StatefulUnit` from a :class:`DataPack` object.

    :param unit: :class:`StatefulUnit` object to be built.
    :param data_pack: The input :class:`DataPack` object.
    :param field_w_mode: list of str or None, if given the mode will be ignored.
    :param mode: One of 'left', 'right', and 'both', to determine the source
            data for building the :class:`VocabularyUnit`.
    :param flatten: Flatten the datapack or not. `True` to organize the
        :class:`DataPack` text as a list, and `False` to organize
        :class:`DataPack` text as a list of list.
    :param verbose: Verbosity.
    :return: A built :class:`StatefulUnit` object.

    """
    corpus = []
    func = corpus.extend if flatten else corpus.append

    if not field_w_mode:
        data_pack.apply_on_text(func, mode=mode, verbose=verbose)
    else:
        for f, m in field_w_mode:
            data_pack.apply_on_field(func, field=f, mode=m, verbose=verbose)

    if verbose:
        description = 'Building ' + unit.__class__.__name__ + \
                      ' from a datapack.'
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit
