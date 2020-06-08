# importing the required libraries
import cv2
import numpy as np
import os

# read recognizer file
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/face_trainer.yml')

# read detector file
face_cascade = cv2.CascadeClassifier('detector/haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

# initiate id counter
ids = 0

names = ['Unknown', 'Rasyid']

cp = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))

while True:
    ret, img = cp.read()
    img = cv2.flip(img,1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img,1.2, 4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
        
        ids, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
         # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            ids = names[ids]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            ids = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(ids), (x+5,y-5), font, 0.7, (255,225,255), 1)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 0.7, (255,225,0), 1)  
    
    out.write(img)
    cv2.imshow('frame', img)
    
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == ord('q'):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cp.release()
cv2.destroyAllWindows()
