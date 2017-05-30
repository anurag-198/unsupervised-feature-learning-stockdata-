import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

mypath = "3classes"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = [mypath + "/" + f for f in onlyfiles]


dataset = []
for files in onlyfiles :
    img = cv2.imread(files)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filename = files.split("/")
    cv2.imwrite("gray503/"  + filename[1] , img)






