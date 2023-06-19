from __future__ import print_function
import cv2 as cv
import argparse
import os
import numpy as np
import time
import copy
import random
import pandas as pd
from IPython.display import display
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import Sequential
from keras import layers, optimizers
from keras.applications import DenseNet121
#from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
#from IPython.display import display
from keras import backend as K
import matplotlib.pyplot as plt
from keras import optimizers

# Находим путь к haar моделям для нахождения лица на изображении. Подробнее про модели: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
cv_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
haar_frontface_model = os.path.join(cv_base_dir, 'data\haarcascade_frontalface_default.xml')

#Грузим .csv файл в датафрейм
pointsData = pd.read_csv("Facial_Keypoints_Dataset/training.csv")
print(pointsData.count())
#Удаляем все строки, где есть хотя бы один NULL
pointsData = pointsData.dropna(how='any',axis=0) 
print(pointsData.count())

#Приводим 1D представление изображения [строкой] в 2D матрицу 96 на 96
pointsData['Image'] = pointsData['Image'].apply(lambda x: np.fromstring(x, dtype= int, sep = ' ').reshape(96,96))

#Делаем копию датафрейма для аугментации данных
pointsData_cp = copy.copy(pointsData)
#Получаем имена столбцов
columns = pointsData_cp.columns[:-1]

#Зеркалим изображение относительно y-координаты
pointsData_cp['Image'] = pointsData_cp['Image'].apply(lambda x: np.flip(x, axis = 1))
#Зеркалим ключевые точки
for i in range(len(columns)):
    if i%2 == 0:    #x-координаты в чётных столбцах
        pointsData_cp[columns[i]] = pointsData_cp[columns[i]].apply(lambda x: 96. - float(x) )

pointsData_aug = np.concatenate((pointsData,pointsData_cp))

pointsData_cp = copy.copy(pointsData)     
#Зеркалим изображение относительно x-координаты
pointsData_cp['Image'] = pointsData_cp['Image'].apply(lambda x: np.flip(x, axis = 0))
#Зеркалим ключевые точки
for i in range(len(columns)):
    if i%2 == 1:    #y-координаты в нечётных столбцах
        pointsData_cp[columns[i]] = pointsData_cp[columns[i]].apply(lambda x: 96. - float(x) )

pointsData_aug = np.concatenate((pointsData_aug, pointsData_cp))

#Слуйчайным образом меняем параметр яркости
pointsData_cp = copy.copy(pointsData)   
pointsData_cp['Image'] = pointsData['Image'].apply(lambda x:np.clip(random.uniform(1, 2) * x, 0.0, 255.0))
pointsData_aug = np.concatenate((pointsData_aug, pointsData_cp))

fig = plt.figure(figsize=(20,20))
fig.tight_layout()
for i in range(16):
    rndm = random.randint(0, pointsData_aug.shape[0])
    ax = fig.add_subplot(4, 4, i + 1)    
    image = plt.imshow(pointsData_aug[rndm][30], cmap = 'gray')
    for j in range(1, 31, 2):
        plt.plot(pointsData_aug[rndm][j-1], pointsData_aug[rndm][j], 'rx')
plt.show()

#Нормализуем изображение, чтобы его пиксели имели вещественные значения от 0 до 1
img = pointsData_aug[:, 30]
img = img/255.

print("Data has been augmented and normalized!")

#Формат данных на входной слой
X = np.empty((len(img), 96, 96, 1)) #Последний аргумент - глубина картинки (кол-во цветов). В нашем случае картинки серые, поэтому 1
for i in range(len(img)):
    X[i,] = np.expand_dims(img[i], axis = 2)
X = np.asarray(X).astype(np.float32)

y = pointsData_aug[:,:30]
y = np.asarray(y).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
   
def res_block(X, filter, stage):
    
  # CONVOLUTIONAL BLOCK
  X_copy = X
  f1 , f2, f3 = filter

  # Main Path
  X = Conv2D(f1, (1,1), strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = MaxPool2D((2,2))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)

  # Short path
  X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
  X_copy = MaxPool2D((2,2))(X_copy)
  X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

  # Add data from main and short paths
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

    
    
  # IDENTITY BLOCK 1
  X_copy = X
    
  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

  # Add both paths together (Note that we feed the original input as is hence the name "identity")
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

    
    
  # IDENTITY BLOCK 2
  X_copy = X

  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

  # Add both paths together (Note that we feed the original input as is hence the name "identity")
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X

input_shape = (96,96,1)

# Input tensor shape
X_input = Input(input_shape)
X = ZeroPadding2D((3,3))(X_input)
X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)
X = res_block(X, filter= [64,64,256], stage= 2)
X = res_block(X, filter= [128,128,512], stage= 3)
X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)
X = Flatten()(X)
X = Dense(4096, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation = 'relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation = 'relu')(X)

model = Model( inputs= X_input, outputs = X)
model.summary()

adam = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) 

# instead of training from scratch, you can load trained model weights
with open('KeyPointDetector.json', 'r') as json_file:
    json_SavedModel = json_file.read()
model = tf.keras.models.model_from_json(json_SavedModel)
model.load_weights('weights.hdf5')
model.compile(loss="mean_squared_error", optimizer = adam, metrics = ['accuracy'])

# Evaluate trained model
result = model.evaluate(X_test,y_test)
print("Accuracy : {}".format(result[1]))

# Make prediction using the testing dataset
print("Making prediction.")
df_predict = model.predict(X_test)

# Print the rmse loss values

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, df_predict))
print("RMSE value : {}".format(rms))

# Convert the predicted values into a dataframe

df_predict= pd.DataFrame(df_predict, columns = columns)
df_predict.head()

# Plot the test images and their predicted keypoints

fig = plt.figure(figsize=(20, 20))

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    # Using squeeze to convert the image shape from (96,96,1) to (96,96)
    plt.imshow(X_test[i].squeeze(),cmap='gray')
    for j in range(1,31,2):
        plt.plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')
plt.show()

pointsData.describe()