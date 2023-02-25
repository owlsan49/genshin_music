# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-02-06
# 完成时间: 2023-xx-xx
# 参考: None
# base: None
# 功能: 从已经标注好的乐谱行(.txt)中切出每一行，作为乐符检测的trainset
from pathlib import Path
from PIL import Image
from data_utils.key_map import reduce_rep

import glob
import os
import cv2

ms_classes = {'0': 'melody_rhythm', '1': 'syllable_lines'}


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def anno2boxes(anno_path):
    """
    function:
        extract yolo form boxes, cls
    :param anno_path:
    :return:
    """
    with open(anno_path, 'r') as f:
        labels = f.readlines()
    labels = [label.split() for label in labels]
    return labels


def crop_line_detect(img_path: str, txt_dir: str, save_dir: str):
    """
    function:
        crop lines from music score preds
    :param img_path:
        source img dir
    :param txt_dir:
        preds dir
    :param save_dir:
    :return:
    """
    img_path = Path(img_path)
    save_dir = Path(save_dir)
    if img_path.is_file():
        ms_img_paths = [str(img_path)]
    else:
        ms_img_paths = glob.glob(str(img_path / '**' / '*.*'), recursive=True)
    # 确保标签文件夹和img文件夹对齐
    ms_img_paths = sorted(x for x in ms_img_paths
                          if x.split('.')[-1].lower() in ['png', 'jpg'])
    ms_anno_paths = glob.glob(str(txt_dir / '**' / '*.txt'), recursive=True)

    for i, mi_path in enumerate(ms_img_paths):
        mi_path = Path(mi_path)
        ms_img = Image.open(str(mi_path))
        ms_anno_path = ms_anno_paths[i]

        with open(ms_anno_path, 'r') as mf:
            ms_lines = [eval(itm) for itm in mf.readlines()]
        ms_lines = sorted(ms_lines, key=lambda x: x[1][1])
        x_labelconf = [[ml[1][1], ml[0]] for ml in ms_lines]
        x_labelconf = reduce_rep(x_labelconf, thr=10)
        ms_lines = [line for i, line in enumerate(ms_lines) if x_labelconf[i] != '##']
        for j, (labelconf, box) in enumerate(ms_lines):
            img_crop = ms_img.crop(box)
            cls = ms_classes[labelconf.split()[0]]
            crop_name = save_dir / f'{mi_path.name.split(".")[0]}_{cls}_{j}.jpg'
            img_crop.save(str(crop_name))


def crop_lines(ms_path: str, save_dir: str):
    """
    function:
        crop img from music scores into rows
    :param ms_path:
    :param save_dir:
    :return:
    """
    ms_path = Path(ms_path)
    save_dir = Path(save_dir)
    ms_img_paths = glob.glob(str(ms_path / '**' / '*.*'), recursive=True)
    # 确保标签文件夹和img文件夹对齐
    ms_img_paths = sorted(x for x in ms_img_paths
                          if x.split('.')[-1].lower() in ['png', 'jpg'])

    ms_anno_paths = sorted(img2label_paths(ms_img_paths))

    for i, mi_path in enumerate(ms_img_paths):
        mi_path = Path(mi_path)
        ms_img = Image.open(str(mi_path))
        shape = ms_img.size
        ms_anno_path = ms_anno_paths[i]
        ms_labels = anno2boxes(ms_anno_path)

        for j, label in enumerate(ms_labels):
            cls, box = label[0], label[1:]
            x0 = (float(box[0]) - float(box[2]) / 2) * shape[0]
            x1 = (float(box[0]) + float(box[2]) / 2) * shape[0]
            y0 = (float(box[1]) - float(box[3]) / 2) * shape[1]
            y1 = (float(box[1]) + float(box[3]) / 2) * shape[1]
            box = (x0, y0, x1, y1)
            img_crop = ms_img.crop(box)
            crop_name = save_dir / f'{mi_path.name.split(".")[0]}_{ms_classes[cls]}_{j}.jpg'
            img_crop.save(str(crop_name))


if __name__ == '__main__':
    # path = r'E:\Programs\Python_Programes\genshin_music_project\datasets\music_score\test\images'
    # save_dir = r'E:\Programs\Python_Programes\genshin_music_project\datasets\rows\val\images'
    # crop_lines(path, save_dir)

    # img_path = r'E:\Programs\Python_Programes\genshin_music_project\datasets\music_score\train\images\人民江山.png'
    # box = (0, 0, 200, 500)
    # ms_img = Image.open(img_path)
    # a = ms_img.crop(box)
    # ms_img.show()
    # a.show()
    pt = '../runs/detect/exp3/ms'


