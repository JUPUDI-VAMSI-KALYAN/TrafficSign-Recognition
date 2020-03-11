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

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: 
        return 'Speed Limit 20 km/h'
    elif classNo == 1: 
        return 'Speed Limit 30 km/h'
    elif classNo == 2: 
        return 'Speed Limit 50 km/h'
    elif classNo == 3: 
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'Yield'
    elif classNo == 10: 
        return 'Stop'
    elif classNo == 11:
        return 'General caution'
    elif classNo == 12:
        return 'Dangerous curve to the left'
    elif classNo == 13:
        return 'Dangerous curve to the right'
    elif classNo == 14:
        return 'Double curve'
    elif classNo == 15:
        return 'Bumpy road'
    elif classNo == 16:
        return 'Slippery road'
    elif classNo == 17:
        return 'Road narrows on the right'
    elif classNo == 18:
        return 'Road work'
    elif classNo == 19:
        return 'Pedestrians'
    elif classNo == 20:
        return 'Children crossing'
    elif classNo == 21:
        return 'Bicycles crossing'
    elif classNo == 22:
        return 'End of all speed and passing limits'
    elif classNo == 23:
        return 'Turn right ahead'
    elif classNo == 24:
        return 'Turn left ahead'
    elif classNo == 25:
        return 'Ahead only'
    elif classNo == 26:
        return 'Go straight or right'
    elif classNo == 27:
        return 'Go straight or left'
    elif classNo == 28:
        return 'Keep right'
    elif classNo == 29:
        return 'Keep left'
    elif classNo == 30:
        return 'No Entry'
    elif classNo == 31:
        return 'No Entry for Heavy Vehicels'

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
        cv2.putText(imgOri,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOri, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOri)
    if getClassName(classIndex)=='Stop' or classIndex==10:
        buzzer()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
