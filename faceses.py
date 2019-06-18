import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd

name = pd.read_csv("names.csv")
name.set_index('name',inplace=True)
print(name)

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')


face_id = len(name)

print("\n [INFO] Look the camera and wait ...")
print(face_id)
count = 0

while(True):

    ret, img = cam.read()
    # img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("training-data/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n Exiting Program ")
cam.release()
cv2.destroyAllWindows()