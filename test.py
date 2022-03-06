# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pickle
import os
import time
import RPi.GPIO as GPIO
PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN,GPIO.OUT)

frameWidth= 640 
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
pickle_in=open("veltech_minor_model.p","rb")
model=pickle.load(pickle_in)


def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img =cv2.equalizeHist(img)
    img = img/255
    return img

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

def buzzer():
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.1)
    GPIO.output(PIN,GPIO.HIGH)
    time.sleep(.10)

    
while True:
    succ, imgOri = cap.read()
    img = np.asarray(imgOri)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOri, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOri, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        cv2.putText(imgOri,str(classIndex)+" "+str(classes_names[classIndex]), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOri, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOri)
    if classes_names[classIndex]=='Stop' or classIndex==10:
        buzzer()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
