import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import tensorflow as tf
import cv2 as cv
import os
from PIL import Image

cv_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
haar_frontface_model = os.path.join(cv_base_dir, 'data\haarcascade_frontalface_default.xml')

def detectFace(frame):
    img = cv.imread(frame)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

    #-- Detect faces
    face_cascade = cv.CascadeClassifier()
    if not face_cascade.load(cv.samples.findFile(haar_frontface_model)):
        print('--(!)Error loading face cascade')
        exit(0)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minSize=(30,30), flags=cv.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        faceROI = gray_img[y:y+h,x:x+w]
    return faceROI

dir = "D:\Project\CK+_Dataset"
i=0

for folder in os.listdir(dir):
    os.mkdir(os.path.join("D:\Project\CK_haared", folder))
    for image_path in os.listdir(os.path.join(dir, folder)):
        path = os.path.join(dir, folder, image_path)
        print(path)
        imag = cv.imread(path, 0)
        imag = detectFace(path)
        imag=np.array(imag)
        im = Image.fromarray(imag)
        im.save("D:\Project\CK_haared\\{folder_name}\\{num}.png".format(folder_name=folder,num=i))
        i=i+1