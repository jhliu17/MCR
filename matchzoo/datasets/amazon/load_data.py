"""Amazon data loader."""

import typing
import matchzoo

from pathlib import Path
from matchzoo.engine.base_task import BaseTask
from matchzoo.helper import logger

from .dataprocess import _read_data


def load_data(read_type, feature_root, *args, **kwargs):
    if read_type == 'all':
        read_func = _read_data
    else:
        raise ValueError(f"Invalid read type {read_type} in (`all`).")
    return _load_data(read_func, feature_root, *args, **kwargs)


def _load_data(
    read_func,
    feature_root,
    cat: str,
    data_root: str,
    stage: str = 'train',
    task: typing.Union[str, BaseTask] = 'ranking',
    filtered: bool = False,
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load Lazada data.

    :param cat: category data.
    :param dataroot: the datapath stores the required data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    logger.info('Loading raw %s data at %s...' % (stage, data_root))
    data_root = Path(data_root)
    prd_file_path = data_root.joinpath('%s.prd.%s' % (cat, stage))
    rvw_file_path = data_root.joinpath('%s.rvw.%s' % (cat, stage))
    rel_file_path = data_root.joinpath('%s.rel.%s' % (cat, stage))
    data_pack = read_func(prd_file_path, rvw_file_path, rel_file_path, task, feature_root)

    if task == 'ranking' or isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif task == 'classification' or isinstance(
            task, matchzoo.tasks.Classification):
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")
