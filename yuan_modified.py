# -*- coding: utf-8 -*-
# 作者: 罗尚
# 创建时间: 2023-01-19
# 完成时间: 2023-xx-xx
# 参考: None
# based on: MaYiming in HeNan XuChang on 2022.4.15

from __future__ import print_function
from pathlib import Path
from argparse import ArgumentParser
import win32con
import win32api
import time

import ctypes, sys

key_map = {
    "0": 49, "1": 50, "2": 51, "3": 52, "4": 53, "5": 54, "6": 55, "7": 56, "8": 57, "9": 58,
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69, "F": 70, "G": 71, "H": 72, "I": 73, "J": 74,
    "K": 75, "L": 76, "M": 77, "N": 78, "O": 79, "P": 80, "Q": 81, "R": 82, "S": 83, "T": 84,
    "U": 85, "V": 86, "W": 87, "X": 88, "Y": 89, "Z": 90
}


def key_down(key):
    """
    函数功能：按下按键
    参    数：key:按键值
    """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), 0, 0)


def key_up(key):
    """
    函数功能：抬起按键
    参    数：key:按键值
    """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), win32con.KEYEVENTF_KEYUP, 0)


def key_press(key):
    """
    函数功能：点击按键（按下并抬起）
    参    数：key:按键值
    """
    key_down(key)
    time.sleep(0.01)
    key_up(key)


def count_note(Note):
    """
    函数功能：为连接的音符数计数
    参    数：Note：相连的音符（中间无空格） 字符串类型, ()算一个音符
    """
    i = 0
    count = 0
    while i < len(Note):
        if Note[i] == '(':
            count += 1
            while 1:
                i += 1
                if Note[i] == ')':
                    i += 1
                    break
        else:
            count += 1
            i += 1
    return count


def play_note(Note, time_div, time_div_div, time_interval):
    """
    函数功能：播放连接的音符
    参    数：Note：相连的音符（中间无空格） 字符串类型
             time_div: 音符时值一次分割
             time_div_div：音符时值二次分割
             time_interval：单个小节的时值
    """
    play_time = time_interval / time_div / time_div_div
    i = 0
    while i < len(Note):
        if Note[i] == '(':
            while 1:
                i += 1
                if Note[i] == ')':
                    time.sleep(play_time)
                    i += 1
                    break
                else:
                    key_press(Note[i])
        elif Note[i].isalpha():
            key_press(Note[i])
            time.sleep(play_time)
            i += 1
        elif Note[i] == '1' or Note[i] == '-':
            time.sleep(play_time)
            i += 1
        else:
            i += 1


def play_music(music, time_interval):
    """
    函数功能：播放曲谱
    参    数：Music：曲谱 字符串类型
             time_interval：单个小节的时值
    """

    music_section = music.split("/")
    for i in range(len(music_section)):
        if music_section[i][-2:] == "  ":
            music_section[i] = music_section[i] + '1'

    for x in music_section:
        Notes = x.split()
        time_div = len(Notes) # 一节中音节组的个数
        for y in Notes:
            time_div_div = count_note(y) # 一个音节组中音符的个数

            play_note(y, time_div, time_div_div, time_interval)


def is_admin():
    try:
        print("Is administration ?")
        return ctypes.windll.shell32.IsUserAnAdmin()
        print("yes.")
    except:
        return False


def play(speed=1.25, txt=None):
    if Path(txt).exists():
        with open(txt, 'r') as f:
            txt = f.readlines()[0]
    if is_admin():
        print("Have fun, begin to play!")
        # 将要运行的代码加到这里hd

        time.sleep(2)
        play_music(txt, speed)
    else:
        print("No permission!")
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--txt', type=str, default=' H- H--- G- H- H--- G- / H- H--- G- H--- Q--- / H- H--- G- H- H--- G- / H--- Q--- W--- E--- / S- D- N B N B S- D- N B N B / S- D- N B N B A- M- N- B- / S- D- N B N B S- D- N B B B / S- D- G- Q- J Q J H G- D- S- D- N B N B S- D- N B N B / S- D- N B N B A- M- N- B- / A- N A S- A S D- S D G Q D G / Q- J- H- G- H--- H- Q- / W- E- H G H G W- E- H G H G / W- E- H G H G Q- J- H- G- / W- E- H G H G W- E- H G H G / W- E- T- Q- J Q Y Y T- E- / W- E- H G H G W- E- H G H G / W- E- H G H G Q- J- H- G- / E E T Y T E E H- Q- E- T- / Y- Y--- T- Y--- ---- / H--- H- - G H- Q- W- E- / H--- H- - G H- G- D- G- / H--- H- - G H- Q- W- E- / E--- W--- Q--- H--- / H--- H- - G H- Q- W- E- / H--- H- - G H- G- G- D- / H--- H- - G G- H- Q- W- / E--- W--- Q--- H--- / Q--- J--- H--- G--- / G- G G D- S- D--- ---- / D- G- H--- W--- J--- / Q--- J- G- H--- ---- / Q--- J--- H--- G--- / G- G G D- S- D--- D- G- / H- H--- H- Q--- W--- / J--- ---- ---- H- Q- W- W--- E- E--- -- E- / T- Y- W- Q- E--- H- Q- / W- W--- E- E--- E- E- / F- E- W- Q- Q--- H- Q- / W- W--- E- E--- -- E- / T- Y- W- Q- E--- H- Q- / F--- E--- W--- Q--- / Q- W- J- G- H--- H- Q- / W- W--- E- E--- -- E- / T- Y- W- Q- E--- H- Q- / W- W--- E- E--- E- E- / E- E- W- Q- Q- Q--- H- Q- / W--- E- E--- -- E- / T- Y- W- Q- E--- H- Q- / E--- E--- W--- Q--- / W- Q- E- T- Y--- ---- /')

    parser.add_argument('--speed', type=float, default=1.25)
    args = parser.parse_args()
    play(args.speed, args.txt)
