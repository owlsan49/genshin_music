# detect
python full_detect.py --source datasets/千本樱.png
--ms-wgt models/weights/ms_best.pth --row-wgt models/weights/row_best.pth

# train music score
python train.py --imgsz 640 --batch-size 8 --epochs 100 --cfg models/yolov5x.yaml
--weights pretrain_model/yolov5x.pt --data ./datasets/music_score/music_line.yaml

# train rows
python train.py --imgsz 660 --batch-size 4 --epochs 100 --cfg models/yolov5x.yaml
--weights pretrain_model/yolov5x.pt --data ./datasets/rows/rows.yaml

