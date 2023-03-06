# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-01-26
# 完成时间: 2023-xx-xx
# 参考: Yolov5
# base: None


from pathlib import Path

import numpy as np

from utils.plots import output_to_target, plot_images
from data_utils.key_map import reduce_rep

import warnings
import os
import torch
import datetime
import cv2
import yaml
import matplotlib.pyplot as plt


class Logger:
    """
    function:
        record, log information in disk. save, restore model weight.
        plot images, etc. The class is for writing down something in
        memory.
    """
    def __init__(self, args, mode='train'):
        # create full save path
        self.args = args
        self.save_dir = Path(args.project) / 'exp'
        self.save_dir = self.__increment_path()

        if mode == 'train':
            # dir for save model
            self.model_dir = self.save_dir / 'weights'
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.best_score = 0
            self.fitness = {'fit': [], 'fit_figure': self.save_dir / 'fitness_per_epoch.jpg'}


            # log hyp and opt
            with open(args.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)
            with open(self.save_dir / 'hyp.yaml', 'w') as f:
                f.write(f'time: {datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")}\n')
                yaml.safe_dump(hyp, f, sort_keys=False)
            with open(self.save_dir / 'opt.yaml', 'w') as f:
                f.write(f'time: {datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")}\n')
                args = {k: str(v) for k, v in vars(args).items()}
                yaml.safe_dump(args, f, sort_keys=False)
        elif mode == 'test':
            self.save_ms = self.save_dir / 'ms'
            self.save_row = self.save_dir / 'rows'
            self.save_ms.mkdir(parents=True, exist_ok=True)
            self.save_row.mkdir(parents=True, exist_ok=True)

    def save_chkp(self, chkp: dir, score):
        last, best = self.model_dir / 'last.pth', self.model_dir / 'best.pth'

        torch.save(chkp, last)
        if score > self.best_score:
            self.best_score = score
            torch.save(chkp, best)

    def save_img(self, img, img_name, suffix='dir'):
        img_file = str(eval(f'self.save_{suffix}') / img_name)
        cv2.imwrite(img_file, img)

    def __increment_path(self, mkdir=True):
        """
        :function:
            Increment file or directory path, i.e. runs/exp2 --> runs/exp2, ... etc.
        :param mkdir:
        :return:
        """
        path = self.save_dir
        if path.exists():
            for n in range(2, 9999):
                p = f'{path}{n}'  # increment path
                if not os.path.exists(p):
                    break
            path = Path(p)

        if mkdir:
            path.mkdir(parents=True, exist_ok=True)  # make directory
        return path

    def plot_val_img(self, im, targets, outputs, paths, i_batch, names):
        plot_images(im, targets, paths, self.save_dir / f'val_batch{i_batch}_labels.jpg', names)
        outp = output_to_target(outputs)
        # for itm in outp[:4]:
        #     print(itm)
        outp = sorted(outp, key=lambda x: x[2])
        outp = sorted(outp, key=lambda x: x[0])
        # for itm in outp:
        #     print(itm)
        if self.args.data.split('/')[-1] == 'music_line.yaml':
            xp = 3
            thr = 40
            # print('AAAAAAAAAA')
        elif self.args.data.split('/')[-1] == 'rows.yaml':
            xp = 2
            thr = 8
            # print('BBBBBBBB')
        else:
            raise Exception(f'No sunch type {self.args.data.split("/")[-1]}')
        outp = reduce_rep(outp, xp=xp, confp=-1, thr=thr)
        # for itm in outp:
        #     print(itm)
        # print()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outp = np.array([otp for otp in outp if otp != '##'])
        # for itm in outp[:4]:
        #     print(itm)
        # exit(0)
        plot_images(im, outp, paths,
                    self.save_dir / f'val_batch{i_batch}_preds.jpg', names)

    def record(self, fitness):
        self.fitness['fit'].append(fitness)
        plt.plot(self.fitness['fit'])
        plt.savefig(self.fitness['fit_figure'])
        plt.clf()

    def save_preds(self, name: str, data, suffix='ms'):
        save_path = (eval(f'self.save_{suffix}') / name).with_suffix('.txt')
        with open(save_path, 'w') as f:
            for itm in data:
                f.write(f'{itm}\n')


if __name__ == '__main__':
    logger = Logger('./runs/train')
