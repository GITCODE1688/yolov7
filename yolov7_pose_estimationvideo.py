from PIL import ImageFont, ImageDraw, Image    # 載入 PIL 相關函式庫
import matplotlib.pyplot as plt
import torch
import cv2
import math
from torchvision import transforms
import numpy as np
import telepot
import os
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' # telegram token
receiver_id = 000000000000 # https://api.telegram.org/bot<TOKEN>/getUpdates
bot = telepot.Bot(token)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt',map_location='cpu')
model = weigths['model']
model = model.float().to(device)
_ = model.eval()


# video_path = 'fall6.mp4'
video_path = './inference/images/FIFA_1080p 00_05_09-00_05_27.mp4'
#pass video to videocapture object
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(1)

#check if videocapture not opened
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

#get video frame width
frame_width = int(cap.get(3))

#get video frame height
frame_height = int(cap.get(4))

#code to write a video
vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
resize_height, resize_width = vid_write_image.shape[:2]
out_video_name = f"{video_path.split('/')[-1].split('.')[0]}"
out = cv2.VideoWriter(f"{out_video_name}_keypoint.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30,
                    (resize_width, resize_height))

#count no of frames
frame_count = 0
#count total fps
total_fps = 0 

#loop until cap opened or video not complete
while(cap.isOpened):
    
    print("Frame {} Processing".format(frame_count))
    frame_count += 1  
    #get frame and success from video capture
    ret, frame = cap.read()
    #if success is true, means frame exist
    if ret:
        
        #store frame
        orig_image = frame
        for i in range(50) :
            out.write(orig_image)
        #convert frame to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (frame_width), stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        for i in range(50) :
            out.write(orig_image)
        
        #convert image data to device
        image = image.float().to(device)  
        
        #convert image to float precision (cpu)
        #image = image.float()
        
        #get predictions
        with torch.no_grad():
            output, _ = model(image)

        #Apply non max suppression
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        im0 = image[0].permute(1, 2, 0) * 255
        im0 = im0.cpu().numpy().astype(np.uint8)
        
        #reshape image format to (BGR)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        for i in range(50) :
            out.write(im0)
        for idx in range(output.shape[0]):
            #plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

            left_shoulder_y= output[idx][23]
            left_shoulder_x= output[idx][22]
            right_shoulder_y= output[idx][26]
            
            left_body_y = output[idx][41]
            left_body_x = output[idx][40]
            right_body_y = output[idx][44]

            len_factor = math.sqrt(((left_shoulder_y - left_body_y)**2 + (left_shoulder_x - left_body_x)**2 ))

            left_foot_y = output[idx][53]
            right_foot_y = output[idx][56]
            
            if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
            #Plotting key points on Image
              cv2.rectangle(im0,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(0, 0, 255),
                  thickness=5,lineType=cv2.LINE_AA)
              
            #   cv2.putText(im0, 'Person Fell down 老人跌倒了', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
              
              fontpath = 'NotoSansTC-Regular.otf'          # 設定字型路徑
              font = ImageFont.truetype(fontpath, 35)      # 設定字型與文字大小
              imgPil = Image.fromarray(im0) 
              draw = ImageDraw.Draw(imgPil)                # 準備開始畫畫
              draw.text((10, 50), 'Person Fell down 人跌倒了', fill=(0, 0, 255), font=font)  # 畫入文字，\n 表示換行
              im0 = np.array(imgPil)

            #   bot.sendMessage(receiver_id, "Person Fall Detected")
              filename = './inference/savedImage.jpg'
              for i in range(10) :
                  cv2.imwrite(filename, im0)
            #   bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
              os.remove(filename)
              cv2.imshow('image', im0)
                            


                           
        #add FPS on top of video
        # cv2.putText(im0, f'FPS: {int(fps)}', (11, 100), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        for i in range(50) :
            cv2.imshow('image', im0)
        out.write(im0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
#avg_fps = total_fps / frame_count
# print(f"Average FPS: {avg_fps:.3f}")
    
# pose_estimation(img)
