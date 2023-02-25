# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-02-09
# 完成时间: 2023-xx-xx
# 参考: None
# base: None
# 功能: 用labelme标注的json格式的图片转为yolo可以训练的格式
import os
import json
import glob

from pathlib import Path


classes = ['_0x', '_0x2', '_0x4',
           '_1x', '_1u', '_1n', '_1u2', '_1n2', '_1x2', '_1u4', '_1n4', '_1x4',
           '_2x', '_2u', '_2n', '_2u2', '_2n2', '_2x2', '_2x4', '_2u4', '_2n4',
           '_3x', '_3u', '_3n', '_3u2', '_3n2', '_3x2', '_3x4', '_3u4', '_3n4',
           '_4x', '_4u', '_4n', '_4u2', '_4n2', '_4x2', '_4x4', '_4u4', '_4n4',
           '_5x', '_5u', '_5n', '_5u2', '_5n2', '_5x2', '_5x4', '_5u4', '_5n4',
           '_6x', '_6u', '_6n', '_6u2', '_6n2', '_6x2', '_6x4', '_6u4', '_6n4',
           '_7x', '_7u', '_7n', '_7u2', '_7n2', '_7x2', '_7x4', '_7u4', '_7n4',
           '_-', '_hx', '_hx2', '_l']

classes_to_label = {classes[i]: str(i) for i in range(len(classes))}


def json_to_yolov5_labels(src_dir, classes_to_label):
    """
    function:
        把json class和坐标 变为yolov5可以训练的形式
    """
    src_dir = Path(src_dir)
    save_path = src_dir.parent / 'labels'

    json_list = glob.glob(str(src_dir / '**' / '*.json'), recursive=True)

    for f in json_list:
        with open(os.path.join(src_dir, f), encoding='utf-8') as jf:
            data = json.load(jf)
            image_name = data['imagePath']
            label_file_path = (save_path / image_name.split('.')[0]).with_suffix('.txt')
            with open(str(label_file_path), 'w') as label_file:
                for d in data['shapes']:
                    name_label = d['label']
                    img_h = data['imageHeight']
                    img_w = data['imageWidth']
                    try:
                        num_label = classes_to_label[name_label]
                    except Exception as e:
                        print(name_label)
                        print(e)
                        exit(0)
                    left_up_point_x, left_up_point_y = d['points'][0]
                    right_down_point_x, right_down_point_y = d['points'][1]
                    x = (left_up_point_x + right_down_point_x) / (2 * img_w)
                    y = (left_up_point_y + right_down_point_y) / (2 * img_h)
                    height = abs((right_down_point_y - left_up_point_y) / img_h)
                    width = abs((right_down_point_x - left_up_point_x) / img_w)
                    label_file.write(num_label+' '+str(x)+' '+str(y)+' '+str(width)+' '+str(height)+'\n')


if __name__ == '__main__':
    # path = r'E:\Programs\Python_Programes\genshin_music_project\datasets\music_score\test\images'
    # ms_class2label = {'melody_rhythm': '0', 'syllable_lines': '1'}
    # json_to_yolov5_labels(path, ms_class2label)

    path = r'E:\Programs\Python_Programes\genshin_music_project\datasets\rows\val\images'
    json_to_yolov5_labels(path, classes_to_label)
    print(classes_to_label)
