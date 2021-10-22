import os

import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm


def get_hierarchical_imageset(root_dir):
    image_set = []

    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.jpg'):
                f_root, f_dir = os.path.split(root)
                image_set.append((f_root, f_dir, f))
    return image_set


def rm_repeat_extraction_task(output_dir, img_list):
    filter_list = []
    for img in img_list:
        im_root, im_dir, im_file = img
        if os.path.exists(os.path.join(output_dir, im_dir, im_file.split('.')[0]+'.npz')):
            continue
        filter_list.append(img)
    return filter_list


def check_bad_image2(img_list):
    for im_root, im_dir, im_file in img_list:
        im_path = os.path.join(im_root, im_dir, im_file)
        print('--')
        print(im_path)
        im = cv2.imread(im_path)


def check_bad_image(img_list):
    bad_list = []
    for im_root, im_dir, im_file in img_list:
        im_path = os.path.join(im_root, im_dir, im_file)
        try:
            im = Image.open(im_path)
            im.verify()
        except:
            print(im_path)
            bad_list.append(im_path)
    print('Bad nums:', len(bad_list))
    return bad_list


def get_hierarchical_file(root_dir, file_type):
    image_set = []

    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(file_type):
                f_root, f_dir = os.path.split(root)
                image_set.append((f_root, f_dir, f))
    return image_set


def stat_roi_num(path):
    file_list = get_hierarchical_file(path, '.npz')
    roi_num_list = []

    for file_args in tqdm(file_list):
        npz_path = os.path.join(*file_args)
        npz = np.load(npz_path)
        feat = npz.get('x', None)
        if feat is not None and len(feat.shape) == 2:
            roi_num_list.append(feat.shape[0])
    
    return len(file_list), np.mean(roi_num_list)
