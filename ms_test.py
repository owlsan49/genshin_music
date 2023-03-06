# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-01-30
# 完成时间: 2023-xx-xx
# 参考: yolov5
# base: None
import numpy as np

from utils.general import (LOGGER, non_max_suppression, scale_coords,
                           check_img_size)
from utils.plots import Annotator, colors
from argument_creater import TestArgCreater
from models.yolo import Model
from logger import Logger
from data_utils.dataloader_creater import LoadImages, Path
import yaml
import torch
import copy
import cv2


class MSTester:

    def __init__(self):
        self.args_creater = TestArgCreater()
        self.args = self.args_creater.parse()
        self.gmp_logger = Logger(self.args, 'test')
        self.imgsz = self.args.ms_sz

        self.data_cfg = self.__read_yaml(self.args.ms_data)
        self.class_names = self.data_cfg['names']
        if self.args.source:
            self.test_source = self.args.source
        else:
            self.test_source = Path(self.data_cfg['path']) / Path(self.data_cfg['test'])
        self.device = self.args.device
        self.conf_thres = self.args.conf_thres
        self.iou_thres = self.args.iou_thres

        # model init
        self.yolov5 = Model(self.args.cfg, nc=self.data_cfg['nc']).to(self.device)
        if self.args.ms_wgt:
            self.__load_chkp(self.args.ms_wgt)

        # dataset init
        gs = 32
        self.imgsz = check_img_size(self.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
        self.test_set = LoadImages(self.test_source, self.imgsz)

    @torch.no_grad()
    def test(self):
        self.yolov5.eval()

        for im, im0s, path in self.test_set:
            im = im.to(self.device).float() / 255.
            if len(im.shape) == 3:
                im = im.unsqueeze(0)

            # im = im.to('cpu').squeeze().numpy()
            # im = im.transpose(1, 2, 0)
            # print(im.shape)
            # cv2.imshow('1', im)
            # cv2.waitKey(0)
            # exit(0)

            # Inference
            # print(im.shape)
            preds, _ = self.yolov5(im)

            # NMS
            preds = non_max_suppression(preds, self.conf_thres, self.iou_thres)
            pred_list = []
            for i, det in enumerate(preds):

                p, im0 = Path(path), im0s.numpy().copy()
                # print(im0.shape, type(im0), im0[:10, :10])
                # im0 = (im.cpu().numpy()[0].copy().transpose(1, 2, 0) * 1).round()
                #
                # im0 = np.ascontiguousarray(im0)
                # print(im0.shape, type(im0), im0[:10, :10])

                annotator = Annotator(im0, line_width=2, font_size=10, example=str(self.class_names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # det--> [ boxes, confidence, classes]

                    # print('aaaaaaa', det[:, :4])
                    # print(im.shape[2:])
                    # gain = min(im.shape[2:][0] / im0.shape[0], im.shape[2:][1] / im0.shape[1])
                    # ratio_pad = [[gain], [0, 0]]
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # det[:, :4] = scale_coords(im.shape, det[:, :4], im.shape).round()
                    # print(im0.shape)
                    # exit(0)
                    for ti, (*xyxy, conf, cls) in enumerate(det):
                        # print('dddddd', xyxy)
                        c = int(cls)  # turn tensor float item into int
                        label = f'{self.class_names[c]} {conf:.2f}'

                        pred_list.append([f'{c} {conf:.2f}', det[ti, :4].tolist()])
                        # print(xyxy)
                        annotator.box_label(xyxy, color=colors(c, True))
                        # annotator.box_label([10, 10, 554, 279], color=(128, 128, 128))

                im0 = annotator.result()

                # print(im0.shape)
                # cv2.imshow('1', im0)
                # cv2.waitKey(0)
                # exit(0)

                self.gmp_logger.save_img(im0, p.name, suffix='ms')
                self.gmp_logger.save_preds(p.name, pred_list)

    @staticmethod
    def __read_yaml(yaml_path):
        with open(yaml_path, errors='ignore') as f:
            yaml_file = yaml.safe_load(f)
        return yaml_file

    def __load_chkp(self, weight):
        # load checkpoint to CPU to avoid CUDA memory leak
        chkp = torch.load(weight, map_location='cpu')
        LOGGER.info(f'Load chkp from {weight}...')
        self.yolov5.load_state_dict(chkp['model'].float().state_dict(), strict=False)


if __name__ == '__main__':
    tester = MSTester()
    tester.test()











