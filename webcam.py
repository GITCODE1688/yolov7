import os
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode

#讀取yolo相關內容
class YOLO(object):
    _defaults = {
        "model_path": '/Users/sonepanpan/Downloads/training/yolov4-custom_best.weights',
        "classes_path": '/Users/sonepanpan/Desktop/Project/YOLOv3-custom-training/model_data/test_classes.txt',
        "cfg_path": '/Users/sonepanpan/Desktop/Project/YOLOv3-custom-training/model_data/yolov4-custom.cfg',
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names,self.classes = self._get_class()
        self._net=cv.dnn.readNet(self.model_path,self.cfg_path)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.weights')
        with open(self.classes_path,'r') as f:
            classes = f.read().splitlines()
        return class_names,classes

    #一次input一個frame
    def detect_img(self, img):
        boxes = []
        confidences = []
        Confidences=[]
        Labels=[]
        class_ids = []

        height, width, _ = img.shape
        blob=cv.dnn.blobFromImage(img, 1/255,(128,128),(0,0,0),swapRB=True,crop=True)
        self._net.setInput(blob)
        output_layers_names = self._net.getUnconnectedOutLayersNames()
        layerOutputs = self._net.forward(output_layers_names)

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    coord=''+str(x)+','+str(y)+','+str(w)+','+str(h)+','
                    print('HI')
                    print(coord)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
                

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv.FONT_HERSHEY_PLAIN

        if len(indexes) > 0:
            indexes=indexes.flatten()

        for i in indexes:
            x, y, w, h = np.array(boxes)[i]
            label = str(self.classes[class_ids[i]])
            Labels.append(label)
            confidence = str(round(confidences[i],2))
            Confidences.append(confidence)
            if label=='qualified':
                color=(255,0,0)
            elif label=='unqualified':
                color=(0,0,255)
            cv.rectangle(img,(x,y),(x+w,y+h),color,4)
            cv.putText(img,label+ ' ' + confidence, (x,y+20), font, 2, color,4)
        return img, Labels, Confidences

#讀取qrcode
def qrcode(frame):
    MyDataBase=['06160512526G011006105892911006','http://onelink.to/tg2k4v']
    if decode(frame)==[]:
        status=''
    else:
        for code in decode(frame):
        #print(code.type)
            status=''
            Content=code.data.decode('utf-8')
            if Content=='06160512526G011006105892911006':
                status='door1'
                color=(0,255,0)
            elif Content=='http://onelink.to/tg2k4v':
                status='door2'
                color=(0,255,0)
            else:
                status='Not-In-System'
                color=(0,0,255)
            pts=np.array([code.polygon],np.int32)
            pts=pts.reshape((-1,1,2))
            cv.polylines(frame,[pts],True,color,5)
            pts2=code.rect
            cv.putText(frame,status,(pts2[0],pts2[1]),cv.FONT_HERSHEY_SIMPLEX,0.9,color,2)
    return frame,status

DeviceName=''
Status='' 

#主程式
if __name__=="__main__":
    # yolo = YOLO()

    # we create the video capture object cap
    cap = cv.VideoCapture(0)
    cap.set(3,640) #width
    cap.set(4,480) #height

    if not cap.isOpened():
        raise IOError("We cannot open webcam")

    while True:
        ret, frame = cap.read()

        #detect qrcode
        Q_image,Q_Code=qrcode(frame)
        if Q_Code!='Not-In-System' and Q_Code!='' and Q_Code!=DeviceName:
            DeviceName=Q_Code        #掃描qrcode顯示並儲存


        # resize our captured frame if we need
        frame = cv.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv.INTER_AREA)

        # detect object on our frame
        r_image, ObjectList, ThresholdList = yolo.detect_img(Q_image)

        if 'qualified' in ObjectList:
            if float(ThresholdList[ObjectList.index('qualified')])>=0.5:
                Status='true'
        elif 'unqualified' in ObjectList:
            if float(ThresholdList[ObjectList.index('unqualified')])>=0.5:
                Status='false'

        print('DeviceName',DeviceName)
        print('Status',Status)
        
        # show us frame with detection
        text_1='Device Name: '+DeviceName
        cv.putText(r_image, text_1, (50, 60), cv.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, 4)
        text_2='Status: '+Status
        cv.putText(r_image, text_2, (50, 90), cv.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, 2)
        cv.imshow("Webcam test", r_image)
        
        #鍵盤操作
        button = cv.waitKey(1) & 0xFF

        #Clear DeviceName and Status Record
        if button == ord('c'):
            DeviceName=''
            Status=''
            continue

        #Save+Reset DeviceName and Status Record
        if DeviceName!='' and  Status!='' and button == ord('s'):
            DeviceName=''
            Status=''
            continue

        #Quit
        if button == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break