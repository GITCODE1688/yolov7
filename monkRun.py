#####
from ast import Try
import os
import cv2
import detect
import shutil
# 取得最後一個資料夾
import os

_lastdir='';
def lastDir():
    rootdir = '/runs'
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            _lastdir = d
                
    return _lastdir;

_lastfile='';
def lastFile(fld):
    rootdir = fld
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isfile(d):
            _lastfile = d
    
    return _lastfile;

if __name__ == '__main__':
    try:
        shutil.rmtree ( "./runs/detect" ) 
    except:
        pass
    
    detect(0)
    