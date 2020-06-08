# importing the required libraries
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Get the face box
detector = cv2.CascadeClassifier('detector/haarcascade_frontalface_default.xml')
    
def Recognize_face(path):
    
    imagepath = [os.path.join(path,ls) for ls in os.listdir(path)]
    
    facesample = list()
    idxs = list()
    
    for im in imagepath:
        imgPIL = Image.open(im).convert('L') #convert to grayscale
        imgNumpy = np.array(imgPIL, 'uint8')
        
        face_id = int(os.path.split(im)[-1].split('.')[1])
        faces = detector.detectMultiScale(imgNumpy)
        
        for (x,y,w,h) in faces:
            facesample.append(imgNumpy[y:y+h, x:x+w])
            idxs.append(face_id)
    return facesample, idxs

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = Recognize_face(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/face_trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
