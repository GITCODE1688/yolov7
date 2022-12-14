## 同步
> https://github.com/GITCODE1688/yolov7.git
## webcam
> --source 0  or --source 1 選擇不同的device


 

## use python version 3.8.8rc1
> use python version 3.8.8rc1
## run test 
> python entry.py   
> python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

## references
### yolo pose
> https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/#YOLOv7-Pose-Code

> yolov7_pose_estimationvideo.py、yolov7pose.py這2支可以做到火柴人識別

> yolov7_pose_estimationvideo.py 可以偵測人員跌倒，但偵測條件看不太懂 , line 96 處