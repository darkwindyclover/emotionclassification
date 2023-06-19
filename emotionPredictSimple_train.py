# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        'D:/Project/FER-2013_Dataset/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
        'D:/Project/FER-2013_Dataset/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

classmodel = Sequential()

classmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
classmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
classmodel.add(MaxPooling2D(pool_size=(2, 2)))
classmodel.add(Dropout(0.25))

classmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
classmodel.add(MaxPooling2D(pool_size=(2, 2)))
classmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
classmodel.add(MaxPooling2D(pool_size=(2, 2)))
classmodel.add(Dropout(0.25))

classmodel.add(Flatten())
classmodel.add(Dense(1024, activation='relu'))
classmodel.add(Dropout(0.5))
classmodel.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

classmodel.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

classmodel_info = classmodel.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

model_json = classmodel.to_json()
with open("classmodel.json", "w") as json_file:
    json_file.write(model_json)

classmodel.save_weights('classmodel.h5')