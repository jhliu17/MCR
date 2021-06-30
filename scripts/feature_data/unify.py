import os
import torch
import numpy as np

from tqdm import tqdm


def get_hierarchical_imageset(root_dir):
    for root, dirs, files in os.walk(root_dir):
        all_files = []
        for f in files:
            if f.endswith('.npz'):
                f_root, f_dir = os.path.split(root)
                all_files.append((f_root, f_dir, f))
        if all_files:
            yield all_files


def read_npz(path):
    try:
        npz = np.load(path, allow_pickle=True)
        if npz['x'].shape[0] == 0:  # some npz file contains zero features...
            print(path, 'is 0 feats')
            return None
    except:
        return None
    data = {}
    keys = npz.keys()
    for k in keys:
        try:
            data[k] = npz[k]
        except:
            continue
    return data


def restore_npy(features_dir):
    for file_list in tqdm(get_hierarchical_imageset(features_dir)):
        npz_list = []
        f_root, f_dir, _ = file_list[0]
        save_root = os.path.join(f_root, f_dir)
        if os.path.exists(os.path.join(save_root, f"{f_dir}.pth")):
            continue
        for args in file_list:
            _, _, f = args
            npz = read_npz(os.path.join(save_root, f))
            if npz is not None:
                npz_list.append(npz)

        if len(npz_list) > 0:
            torch.save(
                npz_list,
                os.path.join(save_root, f"{f_dir}.pth")
            )


def check_pth(features_dir):
    for file_list in tqdm(get_hierarchical_imageset(features_dir)):
        f_root, f_dir, _ = file_list[0]
        save_root = os.path.join(f_root, f_dir)
        pth_f = torch.load(
            os.path.join(save_root, f"{f_dir}.pth")
        )

        for pf in pth_f:
            try:
                p = pf['x'].shape[0]
                if p == 0:
                    raise ValueError
            except:  
                print(os.path.join(save_root, f"{f_dir}.pth"), 'contains valid')


def print_npy(features_dir):
    for file_list in tqdm(get_hierarchical_imageset(features_dir)):
        print('-'*20)
        f_root, f_dir, _ = file_list[0]
        save_root = os.path.join(f_root, f_dir)
        for args in file_list:
            _, _, f = args
            print(save_root, f, os.path.join(save_root, f))


if __name__ == '__main__':
    # check_pth("/home/junhao.jh/dataset/lazada/clothing_prd_y19/set1/features")
    restore_npy("/home/junhao.jh/dataset/amazon18/electronics/set1/features")
    print("Finish!")
