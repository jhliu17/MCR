import os
import torch
import shutil

import torch.distributed as dist
import matchzoo.modeling as modeling_set
from matchzoo.helper import logger
from matchzoo.helper.configure import Configure


def build(configs, stage, ckpt, *args, **kwargs):
    modeling_name = configs.model.modeling
    modeling = getattr(modeling_set, modeling_name, None)
    if modeling is None:
        raise ValueError('Undefined %s modeling' % modeling_name)

    logger.info('Using %s modeling' % modeling_name)
    return modeling(configs, stage, ckpt, *args, **kwargs)


def run(config_path, stage, ckpt, rank=-1):
    # set config
    configs = Configure(config_json_file=config_path)
    torch.manual_seed(configs.train.random_seed)
    torch.cuda.manual_seed(configs.train.random_seed)

    # make logger and checkpoint
    if rank in (0, -1):
        logger.Logger(configs)
        checkpoint_path = configs.train.checkpoint.dir
        _, file_name = os.path.split(config_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if stage == 'train':
            shutil.copyfile(config_path, os.path.join(checkpoint_path, file_name))
    if rank != -1:
        dist.barrier()

    # build model
    model = build(configs, stage, ckpt, rank=rank)

    logger.info('Start modeling at %s stage:' % stage)
    if stage == 'train':
        model.train()
    elif stage == 'dev':
        model.dev()
    else:
        model.test()
