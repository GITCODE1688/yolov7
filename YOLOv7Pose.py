# 在cpu本機執行 .half 要改成 .float      line 27   line67
# select interpreter 要切換到到pyhtone的所在目錄，ex home nb，python 要切換到 anaconda 下才可以正常執行
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
import wget
from os.path import exists 
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
 
 

site_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"
path_to_file = 'yolov7-w6-pose.pt'
file_exists = exists(path_to_file)
if not file_exists:
    file_name = wget.download(site_url)
    print(f'{file_name} 下載完成')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt',map_location='cpu')
model = weigths['model']
model = model.float().to(device)
_ = model.eval()
 
# video_path = '../inference/images/「中秋食月」摸獎影片.mp4'
# video_path = 'D:\workspace\VSCodeProject\YOLO\yolov7\inference\images\「中秋食月」摸獎影片.mp4'
# video_path = 'D:\workspace\VSCodeProject\YOLO\yolov7\inference\images\IMG_1509.MOV'
# video_path = 'D:\workspace\VSCodeProject\YOLO\yolov7\inference\images\DSC02331.jpg'
video_path = r'C:\Users\saxjesus\Documents\vscode_project\yolo\yolov7\inference\images\VID_20221204_133106.mp4'
cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
  print('Error while trying to read video. Please check path again')
 
# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Pass the first frame through `letterbox` function to get the resized image,
# to be used for `VideoWriter` dimensions. Resize by larger side.
vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
resize_height, resize_width = vid_write_image.shape[:2]
 
save_name = f"{video_path.split('/')[-1].split('.')[0]}"
# Define codec and create VideoWriter object .
out = cv2.VideoWriter(f"{save_name}_keypoint.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30,
                    (resize_width, resize_height))
 
 
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

while(cap.isOpened):
  # Capture each frame of the video.
  ret, frame = cap.read()
  if ret:
      orig_image = frame
      image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
      image = letterbox(image, (frame_width), stride=64, auto=True)[0]
      image_ = image.copy()
      image = transforms.ToTensor()(image)
      image = torch.tensor(np.array([image.numpy()]))
      image = image.to(device)
      image = image.float()
 
      # Get the start time.
      start_time = time.time()
      with torch.no_grad():
          output, _ = model(image)
        # Get the end time.
      end_time = time.time()
      # Get the fps.
      fps = 1 / (end_time - start_time)
      # Add fps to total fps.
      total_fps += fps
      # Increment frame count.
      frame_count += 1
 
      output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
      output = output_to_keypoint(output)
      nimg = image[0].permute(1, 2, 0) * 255
      nimg = nimg.cpu().numpy().astype(np.uint8)
      nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
      for idx in range(output.shape[0]):
          plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
 
          # Comment/Uncomment the following lines to show bounding boxes around persons.
          xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
          xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
          cv2.rectangle(
              nimg,
              (int(xmin), int(ymin)),
              (int(xmax), int(ymax)),
              color=(255, 0, 0),
              thickness=1,
              lineType=cv2.LINE_AA
          )
 
      # Write the FPS on the current frame.
      cv2.putText(nimg, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                  1, (0, 255, 0), 2)
      # Convert from BGR to RGB color format.
      cv2.imshow('image', nimg)
      out.write(nimg)
      # Press `q` to exit.
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  else:
      break
# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
