import argparse

from matchzoo.utils.modeling import run
from matchzoo.utils.dist import start_dist, setup_dist, cleanup_dist


def get_args():
    parser = argparse.ArgumentParser('Run model')
    parser.add_argument('--config', required=True,
                        type=str, help='modeling config')
    parser.add_argument('--stage', required=True,
                        type=str, help='modeling stage')

    parser.add_argument('--world_size', default=1, type=int,
                        help='distributed world size')
    parser.add_argument('--ckpt', default=None, type=str, help='checkpoint')
    args = parser.parse_args()
    return args


def distributed_run(rank, world_size, config_path, stage, ckpt):
    """
    :param config: helper.configure, Configure Object
    :param stage: str, ('train', 'dev', 'test')
    """
    setup_dist(rank, world_size)
    run(config_path, stage, ckpt, rank)
    cleanup_dist()


if __name__ == "__main__":
    args = get_args()
    ckpt = args.ckpt if args.ckpt else None

    if args.world_size == 1:
        run(args.config, args.stage, args.ckpt)
    else:
        start_dist(
            distributed_run,
            args=(args.world_size, args.config, args.stage, args.ckpt),
            world_size=args.world_size)
