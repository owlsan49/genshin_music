import os
import glob

from PIL import Image
from pathlib import Path


def img_resize(imgs: str, unisize=(700, 990), save=False):
    imgs = Path(imgs)
    if imgs.is_file():
        img_paths = [imgs]
    else:
        img_paths = glob.glob(str(imgs / '**' / '*.*'), recursive=True)
        img_paths = sorted(x.replace('/', os.sep) for x in img_paths if x.split('.')[-1].lower() in ['png', 'jpg'])

    for i, img_p in enumerate(img_paths):
        img = Image.open(img_p)
        if img is None:
            continue
        img = img.resize(unisize, Image.ANTIALIAS)
        if save:
            img.save(img_p)
        else:
            return img


if __name__ == '__main__':
    pt = '../datasets/music_score/train/images'
    img_resize(pt, save=True)