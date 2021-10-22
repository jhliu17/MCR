# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import ray
import torch
from ray.actor import ActorHandle

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.structures import Instances
from models import add_config
from models.bua.box_regression import BUABoxes
from utils.extract_utils import (get_image_blob, save_bbox, save_roi_features,
                                 save_roi_features_by_bbox)
from utils.progress_bar import ProgressBar
from extract_utils import get_hierarchical_imageset, rm_repeat_extraction_task
sys.path.append('detectron2')


def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3,
                      'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd


def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes,
           'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def generate_npz(extract_mode, pba: ActorHandle, *args):
    if extract_mode == 1:
        save_roi_features(*args)
    elif extract_mode == 2:
        save_bbox(*args)
    elif extract_mode == 3:
        save_roi_features_by_bbox(*args)
    else:
        print('Invalid Extract Mode! ')


@ray.remote(num_gpus=1)
def extract_feat(split_idx, img_list, cfg, args, actor: ActorHandle):
    num_images = len(img_list)
    print('Number of images on split{}: {}.'.format(split_idx, num_images))

    model = DefaultTrainer.build_model(cfg)
    print('model checkpoint path:', cfg.OUTPUT_DIR)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()

    for im_root, im_dir, im_file in img_list:
        if os.path.exists(os.path.join(args.output_dir, im_dir, im_file.split('.')[0]+'.npz')):
            actor.update.remote(1)
            continue

        im_path = os.path.join(im_root, im_dir, im_file)
        output_dir = os.path.join(args.output_dir, im_dir)

        # read image
        try:
            im = cv2.imread(im_path)
        except:
            actor.update.remote(1)
            continue

        if im is None:
            print(im_path, "is illegal!")
            actor.update.remote(1)
            continue
        dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)

        # extract roi features
        if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
            attr_scores = None
            with torch.set_grad_enabled(False):
                # some figures cannot be detected
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model(
                        [dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            # the attr_scores may be None
            if attr_scores is not None:
                attr_scores = [attr_score.cpu() for attr_score in attr_scores]
            generate_npz(1, actor,
                         args, cfg, im_file, im, dataset_dict,
                         boxes, scores, features_pooled, attr_scores, output_dir)
        # extract bbox only
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores = model([dataset_dict])
            boxes = [box.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            generate_npz(2, actor,
                         args, cfg, im_file, im, dataset_dict,
                         boxes, scores, output_dir)
        # extract roi features by bbox
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3:
            if not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz')):
                actor.update.remote(1)
                continue
            bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[
                                    0]+'.npz'))['bbox']) * dataset_dict['im_scale']
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model(
                        [dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if not attr_scores is None:
                attr_scores = [attr_score.data.cpu()
                               for attr_score in attr_scores]
            generate_npz(3, actor,
                         args, cfg, im_file, im, dataset_dict,
                         boxes, scores, features_pooled, attr_scores, output_dir)

        actor.update.remote(1)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int,
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)

    parser.add_argument("--mode", default="caffe",
                        type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str,
                        help='the number of min-max boxes of extractor')

    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="image")
    parser.add_argument('--bbox-dir', dest='bbox_dir',
                        help='directory with bbox',
                        default="bbox")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = setup(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    # Extract features.
    # imglist = os.listdir(args.image_dir)
    imglist = get_hierarchical_imageset(args.image_dir)
    imglist = rm_repeat_extraction_task(args.output_dir, imglist)
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))

    if num_images > 0:
        # set devices
        if args.num_cpus != 0:
            ray.init(num_cpus=args.num_cpus, _lru_evict=True)
        else:
            ray.init(_lru_evict=True)
        num_gpus = min(num_gpus, num_images)
        img_lists = [imglist[i::num_gpus] for i in range(num_gpus)]

        pb = ProgressBar(len(imglist))
        actor = pb.actor

        print('Number of GPUs: {}.'.format(num_gpus))
        extract_feat_list = []
        for i in range(num_gpus):
            extract_feat_list.append(extract_feat.remote(
                i, img_lists[i], cfg, args, actor))

        pb.print_until_done()
        ray.get(extract_feat_list)
        ray.get(actor.get_counter.remote())
        ray.shutdown()


if __name__ == "__main__":
    main()
