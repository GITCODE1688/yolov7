# import wget
# site_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
# file_name = wget.download(site_url)


import shutil
import detect



# del detect/exp
try:
    shutil.rmtree('./runs/detect')
except:
    pass

# inputimg= 'inference\\images\\DSC02328.jpg'     
inputimg= 'inference\\images\\bus.jpg'     
# inputimg= 'inference\\images\\image1.jpg'     
# inputimg='inference\\images\istockphoto-522785736-612x612.jpg'
# inputimg='inference\\images\crowded-mountain-summit-colorado_h.webp'
inputimg = 'inference\\images\\bus.jpg'
inputimg = 'inference\\images\\QRCU3392.JPG'



# inputimg='inference\\images\\「中秋食月」摸獎影片.mp4'

# inputimg = '0'                                              #webcam

str = detect.detect_code(inputimg,False)
print(f"monk function {str}")

x = str.split(',')
for i in x :
    # print(i)
    if i.find('person') > -1 :
    # if i.find('dog') > -1 :
        print(f'偵測人員，共{i.split()[0]}人')
        break




# & C:/Users/monksu/AppData/Local/Programs/Python/Python37/python.exe  detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

# & C:/Users/monksu/AppData/Local/Programs/Python/Python37/python.exe  detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference\images\「中秋食月」摸獎影片.mp4


