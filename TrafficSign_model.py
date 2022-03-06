# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as npy

path = "myData"
labelFile = 'labels.csv'
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 10
imageDimesions = (32, 32, 3)


c = 0
images_list = []
classNo_list = []
myimageList = os.listdir(path)
numberOfClasses = len(myimageList)
print("Arrianging images to a list.....")
for x in range(0, len(myimageList)):
    myPictureList = os.listdir(path + "/" + str(c))
    for y in myPictureList:
        currImgage = cv2.imread(path + "/" + str(c) + "/" + y)
        images_list.append(currImgage)
        classNo_list.append(c)
    c += 1
print(" ")
images = npy.array(images_list)
classNo = npy.array(classNo_list)
print("Images Count : ",len(images))
print("Classes : ",len(classNo))


# split data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


num_of_samples = []
cols = 5
num_classes = numberOfClasses


classes_names = ['Speed Limit 20 km/h','Speed Limit 30 km/h',
           'Speed Limit is 50 km per hour','Speed Limit is 60 km per hour',
           'Speed Limit is 70 km per hour','Speed Limit is 80 km per hour',
           'End of Speed Limit 80 km per hour','Speed Limit is 100 km per hour',
           'Speed Limit 120 km/h','Yield','Stop',
           'General caution','Dangerous curve to the left',
           'Dangerous curve to the right','Double curve',
           'Bumpy road','Slippery road',
           'Road narrows on the right','Road work',
           'Pedestrians','Children crossing',
           'Bicycles crossing','End of all speed and passing limits',
           'Turn right ahead','Turn left ahead',
           'Ahead only','Go straight or right',
           'Go straight or left','Keep right',
           'Keep left','No Entry','No Entry for Heavy Vehicels']



def preprocessing_images(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


X_train = npy.array(list(map(preprocessing_images, X_train)))
X_validation = npy.array(list(map(preprocessing_images, X_val)))
X_test = npy.array(list(map(preprocessing_images, X_test)))
cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train) - 1)])

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()


y_train = to_categorical(y_train, numberOfClasses)
y_validation = to_categorical(y_val, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)
size_Filter_1 = (5, 5)
size_Filter_2 = (3, 3)
pool_size = (2, 2)


def myANNModel():
    model = Sequential()
    model.add((Conv2D(40, size_Filter_1, inpyut_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu')))
    model.add((Conv2D(40, size_Filter_1, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add((Conv2D(20, size_Filter_2, activation='relu')))
    model.add((Conv2D(20, size_Filter_2, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='softmax'))
    # Model Compilation
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


main_model = myANNModel()
print(main_model.summary())
history = main_model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val), epochs=epochs_val, validation_data=(X_validation, y_validation), shuffle=1)


score = main_model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy:', score[1])


Original_image = cv2.imread("1.png")
img = npy.asarray(Original_image)
img = cv2.resize(Original_image, (32, 32))
print("Yes Reshaped")
cv2.imshow('image', Original_image)
img = preprocessing_images(Original_image)
cv2.imshow("Processed Image", img)
print("Preprocessed")
img = img.reshape(1,32, 32, 1)
predictions = main_model.predict(img)
classIndex = main_model.predict_classes(img)
probability_Value = npy.amax(predictions)
if probability_Value > 0.25:
    print("Result")
    print("Detected :",classes_names[classIndex])
    print("Probability :",str(round(probability_Value * 100, 2)) + "%")

#filepath="JK-final.hdf5"
#model.save(filepath)
