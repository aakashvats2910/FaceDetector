import keras
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *

snapshots = []

#capture = cv2.VideoCapture('"Directory of your video here"')

capture = cv2.VideoCapture(0)

print("Info : Press 'q' to exit")

while True:
    ret, frame = capture.read()
    if not ret:
        continue
    cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = cascade_classifier.detectMultiScale(frame, 1.3, 5)
    
    faces = sorted(faces, key=lambda e:e[2]*e[3], reverse=True)
    
    for face in faces:
        x,y,w,h = face
        useful_face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Current Feed : q->exit", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
