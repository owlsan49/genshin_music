# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-02-20
# 参考: None
# base: None
# 功能: 从检测文件或标注文件中获取键盘key值映射
from pathlib import Path
from data_utils.annotation import classes_to_label

import glob
import json

label_map = {
    '_0x': '----', '_1x': 'A---', '_2x': 'S---', '_3x': 'D---', '_4x': 'F---',
    '_5x': 'G---', '_6x': 'H---', '_7x': 'J---',
    '_1u': 'Z---', '_2u': 'X---', '_3u': 'C---', '_4u': 'V---',
    '_5u': 'B---', '_6u': 'N---', '_7u': 'M---',
    '_1n': 'Q---', '_2n': 'W---', '_3n': 'E---', '_4n': 'R---',
    '_5n': 'T---', '_6n': 'Y---', '_7n': 'U---',

    '_0x2': '--', '_1x2': 'A-', '_2x2': 'S-', '_3x2': 'D-', '_4x2': 'F-',
    '_5x2': 'G-', '_6x2': 'H-', '_7x2': 'J-',
    '_1u2': 'Z-', '_2u2': 'X-', '_3u2': 'C-', '_4u2': 'V-',
    '_5u2': 'B-', '_6u2': 'N-', '_7u2': 'M-',
    '_1n2': 'Q-', '_2n2': 'W-', '_3n2': 'E-', '_4n2': 'R-',
    '_5n2': 'T-', '_6n2': 'Y-', '_7n2': 'U-',

    '_0x4': '-', '_1x4': 'A', '_2x4': 'S', '_3x4': 'D', '_4x4': 'F',
    '_5x4': 'G', '_6x4': 'H', '_7x4': 'J',
    '_1u4': 'Z', '_2u4': 'X', '_3u4': 'C', '_4u4': 'V',
    '_5u4': 'B', '_6u4': 'N', '_7u4': 'M',
    '_1n4': 'Q', '_2n4': 'W', '_3n4': 'E', '_4n4': 'R',
    '_5n4': 'T', '_6n4': 'Y', '_7n4': 'U',

    '_-': '----', '_l': '/', '_hx': '--', '_hx2': '-'
}
label2class = {v: k for k, v in classes_to_label.items()}
strains = ['d', 'e', 'f', 'g', 'a', 'b', 'c']


def label_parse(labels: list):
    """
    function:
        map labels to keys
    :param labels:
    :return:
    """
    keys = ''
    for lb in labels:
        keys += f' {label_map[lb]}'
    return keys


def reduce_rep(notes: list, thr=5):
    """
    function:
        reduce anchors which are too closed
    :param notes:
        list: [x or y, label_with_conf] sorted
    :param thr:
    :return:
    """
    if len(notes) >= 1:
        i = 0
        j = 1
        while i < len(notes) and j < len(notes):
            try:
                if abs(float(notes[i][0]) - float(notes[j][0])) < thr:
                    idx = i if notes[i] < notes[j] else j
                    notes[idx] = '##'
                    i += 1 if idx == i else 0
                    j += 1
                else:
                    i = j
                    j += 1
            except:
                print(f'error occur in {notes[i], notes[j]}!')
                exit(0)
    return notes


def strain_change(notes: list, strain='C'):
    aa = ['u', 'x', 'n']
    strain = strain.lower()
    if strain == 'c':
        return notes
    elif strain in strains:
        idx = strains.index(strain)
        strain = idx + 1
        for i, note in enumerate(notes):
            if note in ['_0x', '_0x2', '_0x4', '_-', '_l', '_hx', '_hx2']:
                continue
            else:
                lift = int(note[1]) + strain
                if lift > 7:
                    lift -= 7
                    note = note.replace(note[2], aa[aa.index(note[2])+1])
                notes[i] = f'_{lift}{note[2:]}'
        return notes
    else:
        raise Exception(f'string error, no {strain}')


def xlabel2keys(xlb_src, save_dir, strain_dist=None):
    xlb_src = Path(xlb_src)
    save_dir = Path(save_dir)
    lines_path = glob.glob(str(xlb_src / '*.txt'), recursive=True)
    ms_dist = {}
    # print(lines_path)
    # 把同一首歌对应的每行归拢到一个list
    for lpath in lines_path:
        song_name = Path(lpath).with_suffix('').name.split('_')[0]
        if song_name in ms_dist.keys():
            ms_dist[song_name].append(lpath)
        else:
            ms_dist[song_name] = [lpath]
    # print(ms_dist)
    for name, lines in ms_dist.items():
        strain = strain_dist[name] if strain_dist is not None and name in strain_dist.keys() else 'C'
        kkeys = ''
        for line in lines:
            with open(line, 'r') as f:
                notes = [eval(itm) for itm in f.readlines()]
                notes = reduce_rep(sorted(notes))
                notes = [note for note in notes if note != '##']
                notes = [itm[1].split()[0] for itm in notes]
                notes = strain_change(notes, strain)
            keys = label_parse(notes)
            kkeys += keys
        with open(str(save_dir / name), 'w') as sonf:
            sonf.write(kkeys)


def labelme2xclass(source, save_dir):
    """
    function:
        手动标注的音符map到按键
    :param source:
    :param save_dir:
    :return:
    """
    source = Path(source)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    src_list = glob.glob(str(source / '*.json'), recursive=True)

    for src_path in src_list:
        xlabels = []
        src_path = Path(src_path)
        with open(str(src_path), encoding='utf-8') as jf:
            data = json.load(jf)
            for d in data['shapes']:
                name_label = d['label']
                left_up_point_x = d['points'][0][0]
                xlabels.append([left_up_point_x, name_label])

        save_path = (save_dir / src_path.name).with_suffix('.txt')
        with open(save_path, 'w') as rowf:
            for itm in xlabels:
                rowf.write(f'{itm}\n')


if __name__ == '__main__':
    # src = '../datasets/rows/val/images'
    # dest = '../datasets/rows/val/map_label'
    # labelme2xclass(src, dest)

    source = '../datasets/rows/val/map_label'
    save_dir = '../datasets/rows/val'
    xlabel2keys(source, save_dir, strain_dist={'yosabi1': 'G', '千本樱': 'C'})

    # source = '../runs/detect/exp2/rows'
    # save_dir = '../runs/detect/exp2'
    # xlabel2keys(source, save_dir)

