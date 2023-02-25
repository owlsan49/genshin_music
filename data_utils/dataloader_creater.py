# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-01-19
# 完成时间: 2023-xx-xx
# 参考: Yolov5
# base: None

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset,
                           check_requirements, check_yaml, clean_str, cv2,
                           is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy,
                           xywhn2xyxy, xyxy2xywhn)
# from utils.torch_utils import torch_distributed_zero_first
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import time
import numpy as np
import torch
import glob
import hashlib
import os
import random

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


class MscDataset(Dataset):
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self,
                 path: str,
                 img_size=992,
                 augment=True,
                 hyp=None,
                 batch_size=16,
                 image_weights=False,
                 rect=False,
                 stride=32,
                 pad=0.0,
                 prefix='', ):
        super(MscDataset, self).__init__()
        self.path = Path(path)
        self.augment = augment
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.albumentations = Albumentations() if augment else None

        # retrieve all file under the path
        source_paths = glob.glob(str(self.path / '**' / '*.*'), recursive=True)
        self.img_paths = sorted(
            x.replace('/', os.sep) for x in source_paths if x.split('.')[-1].lower() in ['png', 'jpg'])
        self.label_paths = self.img2label_paths()

        self.hyp = hyp
        self.img_size = img_size

        self.mosaicBorder = [-img_size // 2, -img_size // 2]

        # Check cache
        cache_path = Path(self.label_paths[0]).parent.with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version, 'aaa'  # matches current version
            # print(cache['hash'])
            # print(self.get_hash(self.label_paths + self.img_paths))
            # print(self.get_hash(self.label_paths + self.img_paths))
            assert cache['hash'] == self.get_hash(self.label_paths + self.img_paths), 'bbb'  # identical hash
        except Exception as e:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_paths = [self.img_paths[i] for i in irect]
            self.label_paths = [self.label_paths[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into RAM/VRAM
        self.ims = [None] * n
        gb = 0  # Gigabytes of cached images
        self.im_hw0, self.im_hw = [None] * n, [None] * n

        # multi thread to speed up loading process
        results = ThreadPool(NUM_THREADS).imap(self.load_image, range(n))
        pbar = tqdm(enumerate(results), total=n, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, x in pbar:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = x
            gb += self.ims[i].nbytes
            pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB) ram'
        pbar.close()

    def __getitem__(self, index):
        # index += 1
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        if mosaic:
            # print('11111111111111111111111')
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None
            # cv2.imshow('1', img)
            # cv2.waitKey(0)
            # MixUp augmentation
            if np.random.rand() < hyp['mixup']:
                img, labels = self.mixup(img, labels)

        else:
            # Load image

            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # store img in continuous memory space

        return torch.from_numpy(img), labels_out, self.img_paths[index], shapes

    def __len__(self):
        return len(self.img_paths)

    def img2label_paths(self):
        # Define label paths as a function of image paths
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in self.img_paths]

    def load_image(self, i):
        """
        :function:
            load image from image path
        :param i:
            index of image path
        :return:
        """
        im, ip = self.ims[i], self.img_paths[i]
        if im is None:
            im = cv2.imread(ip)
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]
        else:
            return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def mixup(self, img, labels):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        im2, labels2 = self.load_mosaic(np.random.randint(0, self.n - 1))
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        img = (img * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        return img, labels

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(np.random.uniform(-x, 2 * s + x)) for x in self.mosaicBorder)  # mosaic center x, y
        indices = [index] + np.random.choice(self.indices, size=3).tolist()  # 3 additional image indices
        np.random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image

            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaicBorder)  # border to remove

        return img4, labels4

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."

        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(self.verify_image_label, zip(self.img_paths, self.label_paths, repeat(prefix))),
                        desc=desc,
                        total=len(self.img_paths), )
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}.')
        x['hash'] = self.get_hash(self.label_paths + self.img_paths)
        x['results'] = nf, nm, ne, nc, len(self.img_paths)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def exif_size(self, img):
        # Returns exif-corrected PIL size
        s = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation in [6, 8]:  # rotation 270 or 90
                s = (s[1], s[0])
        except Exception:
            pass

        return s

    def verify_image_label(self, args):
        # Verify one image-label pair
        im_file, lb_file, prefix = args
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
        try:
            # verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = self.exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            assert im.format.lower() in ['png', 'jpg', 'jpeg'], f'invalid image format {im.format}'

            if im.format.lower() in ('jpg', 'jpeg'):
                with open(im_file, 'rb') as f:
                    f.seek(-2, 2)
                    if f.read() != b'\xff\xd9':  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                        msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

            # verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:,
                            1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = segments[i]
                        msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0, 5), dtype=np.float32)
            return im_file, lb, shape, segments, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    def get_hash(self, paths):
        # Returns a single hash value of a list of paths (files or dirs)
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
        h = hashlib.md5(str(size).encode())  # hash sizes
        h.update(''.join(paths).encode())  # hash paths
        return h.hexdigest()  # return hash

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


class LoadImages(Dataset):
    """
    function:
        Load Images from file/dir for test
    """
    def __init__(self, path: str, img_size, stride=32, auto=True):
        super(LoadImages, self).__init__()
        path = Path(path)
        p = str(path)

        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]

        # filtrate images
        self.files = [x for x in files if x.split('.')[-1].lower() in ['jpg', 'png']]
        self.num_files = len(self.files)
        self.img_size = img_size
        self.stride = stride
        self.auto = auto

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img0 = cv2.imread(img_path)
        assert img0 is not None, f'Image Not Found {img_path}'

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.tensor(img), torch.tensor(img0), img_path

    def __len__(self):
        return self.num_files


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def create_dataloader(path: str,
                      imgsz,
                      batch_size,
                      stride=16,
                      single_cls=False,
                      pad=0.0,
                      augment=False,  # augmentation
                      rect=False,
                      image_weights=False,
                      hyp=None,
                      prefix='',
                      shuffle=False,
                      quad=False,
                      cache=None,
                      workers=8,
                      rank=-1):
    msc_dataset = MscDataset(path, imgsz, hyp=hyp, prefix=prefix,
                             stride=int(stride), pad=pad, augment=augment,
                             rect=rect, image_weights=image_weights)
    batch_size = min(batch_size, len(msc_dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    # nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, 8])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(msc_dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(msc_dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=0,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=MscDataset.collate_fn4 if quad else MscDataset.collate_fn), msc_dataset


if __name__ == '__main__':
    from argument_creater import ArgCreater
    import yaml

    argsCreater = ArgCreater()
    args = argsCreater.parse()
    # with open(args.hyp, errors='ignore') as f:
    #     hyps = yaml.safe_load(f)
    # path = r'E:\Programs\Python_Programes\genshin_music_project\datasets\music_score\val'
    # # val_set = MscDataset(path, 992, hyp=hyps, prefix=' val',
    # #                          stride=int(16), pad=0.5, augment=False,
    # #                          rect=True, image_weights=False)
    #
    # val_loader = create_dataloader(path,
    #                                992,
    #                                4,
    #                                hyp=hyps,
    #                                stride=int(32),
    #                                pad=0.5,
    #                                rect=True,
    #                                prefix='val ',
    #                                shuffle=True)
    # print(len(val_loader))
    # for i, btc in enumerate(val_loader):
    #     print(btc)
    lim = LoadImages()
