from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import numpy as np

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

face_id = input('\n enter user id end press <return> ==>  ')

# Get the face box
face_cascade = cv2.CascadeClassifier('detector/haarcascade_frontalface_default.xml')
    
count = 0
while True:
    frame = vs.read()
        
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.2, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3) 
        count += 1
        
        cv2.imwrite('dataset/User.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(100) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == 27:
	    break
    elif count >= 30:
	    break

	# update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

