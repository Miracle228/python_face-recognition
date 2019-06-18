import cv2
import numpy as np
import os
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.csv')
cascadePath = "opencv-files/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0
names = pd.read_csv("names.csv")

name = names.values.tolist()
print(name[1])
# names = ['None', 'name1', 'name2', 'name3', 'name4', 'name5']


cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

email = False

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = name[id]
            confidence = round(100 - confidence)
            if confidence > 70:
                email = True
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if email == True:
    print( "da da ya")
else :
    print("unknown")

# Do a bit of cleanup
print("\n  Exiting Program ")
cam.release()
cv2.destroyAllWindows()