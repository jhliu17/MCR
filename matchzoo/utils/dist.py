import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_dist(rank, world_size, backend='nccl', port='12355'):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)

        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_dist():
    dist.destroy_process_group()


def start_dist(func, args, world_size):
    mp.spawn(func,
             args=args,
             nprocs=world_size,
             join=True)
