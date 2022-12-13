import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts














device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location='cpu')
model = weigths['model']
model = model.float().to(device)
_ = model.float().eval()

# if torch.cuda.is_available():
#     model.float().to(device)

image = cv2.imread('./inference/images/DSC02328.jpg')
# image = cv2.imread('./inference/images/img_0650.jpg')
image = letterbox(image, 960, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))

if torch.cuda.is_available():
    image = image.float().to(device)   
output, _ = model(image)


output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
with torch.no_grad():
    output = output_to_keypoint(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    # print ('idx--------------------------------------------')
    # print (idx)
    # print('output------------------------------------------')
    # print(output)
    # print('output[idx, 7:]---------------------------------')
    # print(output[idx, 7:])
    # print('output[idx, 7:].T-------------------------------')
    # print(output[idx, 7:].T)
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

cv2.imwrite('./inference/images/detect.jpg',nimg)
cv2.imshow('image', nimg)

# %matplotlib inline
# plt.savefig(nimg)
# plt.figure(figsize=(8,8))
# plt.axis('off')
# plt.imshow(nimg)
# plt.show()
cv2.waitKey(0)