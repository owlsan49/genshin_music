# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-01-19
# 完成时间: 2023-xx-xx
# 参考: Yolov5
# base: None

from utils.general import (LOGGER, non_max_suppression, scale_coords, xywh2xyxy,
                           intersect_dicts, colorstr, one_cycle, increment_path, check_img_size)
from utils.metrics import ap_per_class, box_iou, fitness
from utils.loss import ComputeLoss
from utils.torch_utils import ModelEMA
from utils.autoanchor import check_anchors
from utils.callbacks import Callbacks
from argument_creater import ArgCreater
from data_utils.dataloader_creater import create_dataloader
# from data_utils.dataloaders import create_dataloader
from models.yolo import Model
from tqdm import tqdm
from pathlib import Path
from logger import Logger
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

import numpy as np
import math
import time
import torch.nn as nn
import torch
import yaml
import os


class Trainer:

    def __init__(self):
        # var init
        self.argsCreater = ArgCreater()
        self.args = self.argsCreater.parse()
        hyps = self.read_yaml(self.args.hyp)
        self.hyp = hyps
        self.data_cfg = self.read_yaml(self.args.data)
        data_path = Path(self.data_cfg['path'])
        self.train_path = data_path / self.data_cfg['train']
        self.val_path = data_path / self.data_cfg['val']
        self.device = self.args.device
        self.imgsz = self.args.imgsz
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs

        # logger init
        self.gmp_logger = Logger(self.args)

        # model init
        self.yolov5 = Model(self.args.cfg, nc=self.data_cfg['nc']).to(self.device)
        if self.args.weights:
            self.__load_chkp()
        self.yolov5.hyp = hyps
        self.yolov5.names = self.data_cfg['names']

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in self.yolov5.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        self.optimizer = torch.optim.SGD(g[2], lr=hyps['lr0'], momentum=hyps['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': g[0],
                                        'weight_decay': hyps['weight_decay']})  # add g0 with weight_decay
        self.optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
        LOGGER.info(f"{colorstr('optimizer:')} {type(self.optimizer).__name__} with parameter groups "
                    f"{len(g[1])} weight (no decay), {len(g[0])} weight, {len(g[2])} bias")
        del g

        # EMA
        self.ema = ModelEMA(self.yolov5)

        if self.args.cos_lr:
            lf = one_cycle(1, hyps['lrf'], self.args.epochs)  # cosine 1->hyp['lrf']
        else:
            lf = lambda x: (1 - x / self.args.epochs) * (1.0 - hyps['lrf']) + hyps['lrf']  # linear
        self.lf = lf
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        # loss func and optim init
        self.loss_func = ComputeLoss(self.yolov5)
        self.scaler = torch.cuda.amp.GradScaler()

        # Image size
        gs = 32
        self.imgsz = self.check_img_size(gs, floor=gs * 2)  # verify imgsz is gs-multiple
        self.train_loader, train_set = create_dataloader(self.train_path,
                                                         self.imgsz,
                                                         self.batch_size,
                                                         stride=32,
                                                         rect=False,
                                                         hyp=hyps, prefix='Train ',
                                                         shuffle=True)

        self.val_loader, _ = create_dataloader(self.val_path,
                                               self.imgsz,
                                               self.batch_size * 2,
                                               hyp=hyps,
                                               stride=int(gs),
                                               pad=0.5,
                                               rect=True,
                                               prefix='val ',
                                               shuffle=False)
        # this process is important to change detect layer
        if not self.args.noautoanchor:
            check_anchors(train_set, model=self.yolov5, thr=hyps['anchor_t'], imgsz=self.imgsz)
        self.yolov5.half().float()

    def check_img_size(self, s=32, floor=0):
        # Verify image size is a multiple of stride s in each dimension
        if isinstance(self.imgsz, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(self.imgsz, int(s)), floor)
        else:  # list i.e. img_size=[640, 480]
            imgsz = list(self.imgsz)  # convert to list if tuple
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in imgsz]
        if new_size != self.imgsz:
            LOGGER.warning(f'WARNING: --img-size {self.imgsz} must be multiple of max stride {s}, updating to {new_size}')
        return new_size

    def make_divisible(self, x, divisor):
        # Returns nearest x divisible by divisor
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # to int
        return math.ceil(x / divisor) * divisor

    def train(self):
        num_itg = 0  # number integrated batches (since train start)
        num_batch = len(self.train_loader)

        # number of warmup iterations, max(3 epochs, 100 iterations)
        num_warmup = max(round(3 * num_batch), 100)
        for epoch in range(0, self.epochs):
            self.yolov5.train()
            LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem',
                                               'box', 'obj', 'cls',
                                               'labels', 'img_size'))

            pbar = tqdm(enumerate(self.train_loader), total=num_batch,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            mloss = torch.zeros(3, device=self.device)  # mean losses

            self.optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:
                num_itg += i
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.
                targets = targets.to(self.device)

                # Warmup
                if num_itg <= num_warmup:
                    xi = [0, num_warmup]  # x interp
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(num_itg, xi,
                                            [self.hyp['warmup_bias_lr']
                                             if j == 0 else 0.0,
                                             x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(num_itg, xi,
                                                      [self.hyp['warmup_momentum'],
                                                       self.hyp['momentum']])

                # predict and compute loss
                # automatic mixed precision to speed up prop
                with torch.cuda.amp.autocast():
                    preds = self.yolov5(imgs)
                    loss, loss_items = self.loss_func(preds, targets)
                self.scaler.scale(loss).backward()

                # optimize
                self.scaler.step(self.optimizer)  # optimizer.step
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.yolov5)

                # show information
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{self.epochs - 1}', mem, *mloss,
                                      targets.shape[0], imgs.shape[-1]))

            # Scheduler
            lr = [x['lr'] for x in self.optimizer.param_groups]  # for loggers
            self.scheduler.step()

            self.ema.update_attr(self.yolov5, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == self.epochs) # or self.stopper.possible_stop

            # evaluate
            results, maps, _ = self.evaluate()
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            self.gmp_logger.record(fi)
            best_fitness = 0
            if fi > best_fitness:
                best_fitness = fi

            # save
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(self.yolov5).half(),
                'optimizer': self.optimizer.state_dict(),
            }
            self.gmp_logger.save_chkp(ckpt, fi)
            pbar.close()

    @torch.no_grad()
    def evaluate(self):
        self.yolov5.eval()

        jdict, stats, ap, ap_class = [], [], [], []
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        names = {k: v for k, v in enumerate(self.yolov5.names
                                            if hasattr(self.yolov5, 'names')
                                            else self.yolov5.module.names)}
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                    desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss = torch.zeros(3, device=self.device)
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for eva_i, (im, targets, paths, shapes) in pbar:
            im = im.to(self.device, non_blocking=True).float() / 255.
            targets = targets.to(self.device, non_blocking=True)
            nb, _, height, width = im.shape  # batch size, channels, height, width

            with torch.cuda.amp.autocast():
                out, train_out = self.yolov5(im)
                loss += self.loss_func([x.float() for x in train_out], targets)[1]  # box, obj, cls
            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]  # for autolabelling
            out = non_max_suppression(out, 0.001, 0.6, labels=[], multi_label=True, agnostic=False)

            # Metrics
            iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()
            seen = 0
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                path, shape = Path(paths[si]), shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=self.device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((3, 0), device=self.device)))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = self.process_batch(predn, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Plot images
            if eva_i < 3:
                self.gmp_logger.plot_val_img(im, targets, out, paths, eva_i, names)  # labels
        pbar.close()

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = \
                ap_per_class(*stats, plot=True,
                             save_dir=self.gmp_logger.save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=self.data_cfg['nc'])  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # Print speeds
        t = tuple(x / seen * 1E3 for x in [0.0, 0.0, 0.0])  # speeds per image

        # Return results
        maps = np.zeros(self.data_cfg['nc']) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map50, map, *(loss.cpu() / len(self.val_loader)).tolist()), maps, t

    def __load_chkp(self):
        # load checkpoint to CPU to avoid CUDA memory leak
        chkp = torch.load(self.args.weights, map_location='cpu')
        LOGGER.info(f'Load chkp from {self.args.weights}...')
        model_weights = intersect_dicts(chkp['model'].float().state_dict(),
                                        self.yolov5.state_dict())
        self.yolov5.load_state_dict(model_weights, strict=False)

    @staticmethod
    def process_batch(detections, labels, iouv):
        """
        function:
            Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    @staticmethod
    def read_yaml(yaml_path):
        with open(yaml_path, errors='ignore') as f:
            yaml_file = yaml.safe_load(f)
        return yaml_file


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    # args_creater = ArgCreater()
    # opt = args_creater.parse()
    # opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True))
    # train(opt.hyp, opt, 'cuda', Callbacks())
