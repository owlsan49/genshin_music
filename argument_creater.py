# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-01-19
# 完成时间: 2023-xx-xx
# 参考: Yolov5
# base: None

import argparse
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class ArgCreater:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # model
        self.parser.add_argument('--weights', type=str, default=ROOT / 'pretrain_model/yolov5x.pt',
                                 help='initial weights path')
        self.parser.add_argument('--cfg', type=str, default='./models/yolov5x.yaml', help='model.yaml path')
        self.parser.add_argument('--data', type=str, default=ROOT / 'datasets/music_line.yaml',
                                 help='dataset.yaml path')
        self.parser.add_argument('--hyp', type=str, default=ROOT / 'data_utils/hyps/hyp.scratch-low.yaml',
                                 help='hyperparameters path')

        # train process
        self.parser.add_argument('--epochs', type=int, default=5)
        self.parser.add_argument('--batch-size', type=int, default=8,
                                 help='total batch size for all GPUs, -1 for autobatch')
        self.parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                                 help='train, val image size (pixels)')

        #
        self.parser.add_argument('--rect', action='store_true', help='rectangular training')
        self.parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        self.parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        self.parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        self.parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
        self.parser.add_argument('--noplots', action='store_true', help='save no plot files')
        self.parser.add_argument('--evolve', type=int, nargs='?', const=300,
                                 help='evolve hyperparameters for x generations')
        self.parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        self.parser.add_argument('--cache', type=str, nargs='?', const='ram',
                                 help='--cache images in "ram" (default) or "disk"')
        self.parser.add_argument('--image-weights', action='store_true',
                                 help='use weighted image selection for training')
        self.parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        self.parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        self.parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD',
                                 help='optimizer')
        self.parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        self.parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')

        # save process
        self.parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
        self.parser.add_argument('--name', default='exp2', help='save to project/name')
        self.parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

        self.parser.add_argument('--quad', action='store_true', help='quad dataloader')
        self.parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
        self.parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        self.parser.add_argument('--patience', type=int, default=100,
                                 help='EarlyStopping patience (epochs without improvement)')
        self.parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                                 help='Freeze layers: backbone=10, first3=0 1 2')
        self.parser.add_argument('--save-period', type=int, default=-1,
                                 help='Save checkpoint every x epochs (disabled if < 1)')
        self.parser.add_argument('--local_rank', type=int, default=-1,
                                 help='Automatic DDP Multi-GPU argument, do not modify')

        # Weights & Biases arguments
        self.parser.add_argument('--entity', default=None, help='W&B: Entity')
        self.parser.add_argument('--upload_dataset', nargs='?', const=True, default=False,
                                 help='W&B: Upload data, "val" option')
        self.parser.add_argument('--bbox_interval', type=int, default=-1,
                                 help='W&B: Set bounding-box image logging interval')
        self.parser.add_argument('--artifact_alias', type=str, default='latest',
                                 help='W&B: Version of dataset artifact to use')

    def parse(self):
        return self.parser.parse_args()


class TestArgCreater:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data
        self.parser.add_argument('--ms-data', type=str, default=ROOT / 'datasets/music_score/music_line.yaml',
                                 help='dataset.yaml path')
        self.parser.add_argument('--row-data', type=str, default=ROOT / 'datasets/rows/rows.yaml',
                                 help='dataset.yaml path')
        self.parser.add_argument('--ms-sz', type=int, default=700,
                                 help='train, val image size (pixels)')
        self.parser.add_argument('--row-sz', type=int, default=660,
                                 help='row image size (pixels)')
        self.parser.add_argument('--source', type=str, default=None, help='test source')

        # model
        self.parser.add_argument('--cfg', type=str, default='./models/yolov5x.yaml', help='model.yaml path')
        self.parser.add_argument('--ms-wgt', type=str, default=ROOT / 'runs/train/ms/exp2/weights/best.pth',
                                 help='initial weights path')
        self.parser.add_argument('--row-wgt', type=str, default=ROOT / 'runs/train/rows/exp2/weights/best.pth',
                                 help='initial weights path')

        # detect hyp
        self.parser.add_argument('--conf-thres', type=float, default=0.25, help='NMS confidence threshold')
        self.parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')

        # system
        self.parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

        # save
        self.parser.add_argument('--project', default=ROOT / 'runs/detect', help='save to project/name')

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':
    # FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[0]
    # if str(ROOT) not in sys.path:
    #     sys.path.append(str(ROOT))  # add ROOT to PATH
    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
    print(FILE.parents[0])
