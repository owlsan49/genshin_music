<<<<<<< HEAD
# 介绍

这是一个

# 使用

使用

=======
# 介绍

​	项目基于yolov5，检测简谱中的乐符并生成相应的键值映射，使用自动演奏程序可以在原神中自动演奏曲子。

![image-20230225163713806](../../../文件/笔记本/image/image-20230225163713806.png)

# 使用

### 1.检测映射

下载完项目文件后

下载训练好的权重文件https://drive.google.com/drive/folders/12FSuGoN79dR93I2TFhBkwirfsyAfCohS?usp=sharing，放到models下的weights文件夹下（下载.pth文件就行）

![image-20230225193149835](../../../文件/笔记本/image/image-20230225193149835.png)

使用命令生成按键映射：

```
python full_detect.py --source datasets/千本樱.png --ms-wgt models/weights/ms_best.pth --row-wgt models/weights/row_best.pth
```

--source是简谱图片源或包含多个原图片的文件夹

*键值映射结果在runs/detect/exp?/下：*

![image-20230225194818744](../../../文件/笔记本/image/image-20230225194818744.png)

###2.自动演奏

有曲子的键值映射就可以直接使用自动演奏



打开原神的风花琴或须弥琴：

使用下面的命令就可以实现自动演奏（使用命令--speed可以控制演奏速度)：

（这一步需要开管理员cmd，软件控制键盘映射需要，如果不会可以用下面的直接方法）

```
python yuan_modified.py --txt runs/detect/exp/千本樱
```

--txt是刚刚生成的键值映射文件位置（本例在runs/detect/exp/千本樱）



或者修改yuan_modified.py的__main__进程中直接修改--txt的default值就，就可以直接运行。

确认权限后，调换到原神游戏界面就会自动演奏



# 训练

用同样的思路也可以检测五线谱（但是我看不懂乐符

如果想用自己的数据训练可以使用train.py这个文件

我先用模型切割出每一行，再对每一行单独检测乐符

具体命令如下：

```
# train music score
python train.py --imgsz 640 --batch-size 8 --epochs 100 --cfg models/yolov5x.yaml
--weights pretrain_model/yolov5x.pt --data ./datasets/music_score/music_line.yaml

# train rows
python train.py --imgsz 660 --batch-size 4 --epochs 100 --cfg models/yolov5x.yaml
--weights pretrain_model/yolov5x.pt --data ./datasets/rows/rows.yaml
```

yolov5x.pt的预训练模型在这里下载[github.com](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)

另外我也提供自己手动标注的数据集，从这里下载https://drive.google.com/drive/folders/12FSuGoN79dR93I2TFhBkwirfsyAfCohS?usp=sharing（两个文件夹）



###编码细节

```
['_0x', '_0x2', '_0x4',
 '_1x', '_1u', '_1n', '_1u2', '_1n2', '_1x2', '_1u4', '_1n4', '_1x4',
 '_2x', '_2u', '_2n', '_2u2', '_2n2', '_2x2', '_2x4', '_2u4', '_2n4',
 '_3x', '_3u', '_3n', '_3u2', '_3n2', '_3x2', '_3x4', '_3u4', '_3n4',
 '_4x', '_4u', '_4n', '_4u2', '_4n2', '_4x2', '_4x4', '_4u4', '_4n4',
 '_5x', '_5u', '_5n', '_5u2', '_5n2', '_5x2', '_5x4', '_5u4', '_5n4',
 '_6x', '_6u', '_6n', '_6u2', '_6n2', '_6x2', '_6x4', '_6u4', '_6n4',
 '_7x', '_7u', '_7n', '_7u2', '_7n2', '_7x2', '_7x4', '_7u4', '_7n4',
 '_-', '_hx', '_hx2', '_l']
```

这是我使用的编码，例如'\_0x2'表示0八分音符，'\_2u2'表示2低音八分音符，'_7n4'表示7高音16分音符

'_hx'表示附点音符，'\_hx2'表示附点八分音符，'\_l'表示小节线，其他的以此类推。

>>>>>>> 01551db (first commit)
