import os
import torch
import numpy as np

from matchzoo.engine.base_callback import BaseCallback
from prefetch_generator import BackgroundGenerator
from matchzoo.helper import logger


class LoadImage(BaseCallback):
    """
    Load related Image Data
    """

    def __init__(
        self,
        feat_dir: str,
        feat_size: int,
        max_roi_num: int,
        img_min_length: int,
        max_prefetch: int = 8
    ):
        self.feat_dir = feat_dir
        self.feat_size = feat_size
        self.max_roi_num = max_roi_num
        self.img_min_length = img_min_length
        self.max_prefetch = max_prefetch

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        "Load related image features, instance without image will be padded"
        id_left = x['id_left']
        id_right = x['id_right']

        max_pref = min(self.max_prefetch, len(id_left))
        image_left, image_left_length = self.load_images(id_left, max_pref)
        image_right, image_right_length = self.load_images(id_right, max_pref)

        x['image_left'] = image_left
        x['image_right'] = image_right
        x['image_left_length'] = image_left_length
        x['image_right_length'] = image_right_length

    def load_image_per_instance(self, dir_names):
        for dir_name in dir_names:
            files_dir = os.path.join(self.feat_dir, dir_name)
            pth_path = os.path.join(files_dir, f"{dir_name}.pth")
            if not os.path.exists(pth_path):
                yield [self._placeholder_img()]
            else:
                try:
                    npz_files = torch.load(pth_path)
                except:
                    logger.info(f"{pth_path} is invalid.")
                    npz_files = []

                feat = []
                for n in npz_files:
                    f = n['x'][:self.max_roi_num]
                    if f.shape[0] == 0:
                        continue
                    feat.append(f)
                if len(feat) > 0:
                    yield feat
                else:
                    logger.info("%s feats contains 0 feature, plz check" % pth_path)
                    yield [self._placeholder_img()]
        
    def load_images(self, dir_names, max_pref):
        image = []
        length = []
        for npz_list in BackgroundGenerator(self.load_image_per_instance(dir_names), max_prefetch=max_pref):
            # concate feats
            feats = np.concatenate(npz_list, axis=0)
            image.append(feats)
            length.append(feats.shape[0])
        return image, length

    def _placeholder_img(self):
        placeholder_f: np.array = np.full([self.img_min_length, self.feat_size], 0, dtype=np.float)
        return placeholder_f
