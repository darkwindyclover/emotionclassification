import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import pandas as pd
from keras import layers, optimizers
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.applications import DenseNet121
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt

cv_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
haar_frontface_model = os.path.join(cv_base_dir, 'data\haarcascade_frontalface_default.xml')

def detectFace(frame):
    img = cv.imread(frame)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Конвертируем изображение в оттенки серого для работы классификатора
    #gray_img = cv.equalizeHist(gray_img) # Уравниваем гистограмму, получаем более контрастное изображение (можно закомментить)

    #-- Detect faces
    face_cascade = cv.CascadeClassifier()
    if not face_cascade.load(cv.samples.findFile(haar_frontface_model)):
        print('--(!)Error loading face cascade')
        exit(0)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minSize=(30,30), flags=cv.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        faceROI = gray_img[y:y+h,x:x+w]
    return faceROI

dir = "D:\Project\CK+_Dataset\\fear"
all_images = []
for image_path in os.listdir(dir):
    path = os.path.join(dir, image_path)
    imag = cv.imread(path, 0)
    imag = detectFace(path)
    imag = cv.resize(imag, (96,96))
    imag=np.array(imag)
    #imag=imag.astype('float32')
    all_images.append(imag)
x_test_fer = np.array(all_images)
x_test_fer = x_test_fer/255.

X = np.empty((len(x_test_fer), 96, 96, 1)) #Последний аргумент - глубина картинки (кол-во цветов). В нашем случае картинки серые, поэтому 1
for i in range(len(x_test_fer)):
    X[i,] = np.expand_dims(x_test_fer[i], axis = 2)
X = np.asarray(X).astype(np.float32)

print(x_test_fer.shape)
print(x_test_fer)

adam = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

with open('KeyPointDetector.json', 'r') as json_file:
    json_SavedModel = json_file.read()
model = tf.keras.models.model_from_json(json_SavedModel)
model.load_weights('weights.hdf5')
model.compile(loss="mean_squared_error", optimizer = adam, metrics = ['accuracy'])

#result = model.evaluate(X_test,y_test)
#print("Accuracy : {}".format(result[1]))

# Make prediction using the testing dataset
print("Making prediction.")
df_predict = model.predict(X)

# Print the rmse loss values

from sklearn.metrics import mean_squared_error
from math import sqrt

#rms = sqrt(mean_squared_error(y_test, df_predict))
#print("RMSE value : {}".format(rms))

# Convert the predicted values into a dataframe

facialpoints_df = pd.read_csv("Facial_Keypoints_Dataset/training.csv")
columns = facialpoints_df.columns[:-1]
df_predict= pd.DataFrame(df_predict, columns = columns)
df_predict.head()

# Plot the test images and their predicted keypoints

fig = plt.figure(figsize=(96, 96))

for i in range(25):
    ax = fig.add_subplot(5, 5, i + 1)
    # Using squeeze to convert the image shape from (96,96,1) to (96,96)
    plt.imshow(x_test_fer[i].squeeze(),cmap='gray')
    for j in range(1,31,2):
        plt.plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')
plt.show()