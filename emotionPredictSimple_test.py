import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import tensorflow as tf
import cv2 as cv
import os

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

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

datapath = 'D:\Project\CK_haared'
#datapath = 'D:\\Project\\FER-2013_Dataset\\test'

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        datapath,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# do prediction on test data
# predictions = emotion_model.predict(test_generator)

predictions = np.array([])
labels = np.array([])

i = 0
for x,y in test_generator:
    predictions = np.concatenate([predictions, np.argmax(emotion_model.predict(x),axis=-1)])
    labels = np.concatenate([labels, np.argmax(y,axis=-1)])
    i += 1
    if i > 100:  # this if-break statement reduces the running time.
        break
ConfusionMatrixDisplay(
    confusion_matrix=tf.math.confusion_matrix(
        labels=labels, predictions=predictions)
    .numpy(), display_labels=emotion_dict).plot(cmap=plt.cm.Blues)
plt.show()
print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())

print("-----------------------------")
print(classification_report(labels, predictions))
