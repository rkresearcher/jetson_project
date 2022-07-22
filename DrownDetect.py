import object_detection as cv
from object_detection import draw_bbox
import cv2
import time
import numpy as np
#for PiCamera
#from picamera Import PiCamera
#camera = PiCamera
#camera.start_preview()
# open webcam
webcam = cv2.VideoCapture('data/Wavepool Lifeguard Rescue 7 - Spot the Drowning!-YgUglYhVSkk.mp4')
age = 0
newb = 0
ret, frame = webcam.read()
if not webcam.isOpened():
    print("Could not open webcam")

t0 = time.time() #gives time in seconds after 1970

#variable dcount stands for how many seconds the person has been standing still for
centre0 = np.zeros(2)
isDrowning = False

#this loop happens approximately every 1 second, so if a person doesn't move,
#or moves very little for 10seconds, we can say they are drowning

#loop through frames
while webcam.isOpened():
    bbox1, label1, conf1,p_id1 = cv.detect_common_objects(frame)
    # read frame from webcam
    status, frame = webcam.read()        
    bbox, label, conf,p_id = cv.detect_common_objects(frame)

    newb=time.time()
    act = newb-age
    age = newb
       # from this point I have to do filtering out based on p_id, and code will use FSM based on p_id, so that every object get correct detecion i.e., Drowning or Nowmal 
    j = 0
    for i in bbox:
        if label[j] == 'person':
           if 0<i[3]<=250:
                
                isDrowning = True
                out = draw_bbox(frame, i, label[j], conf[j],isDrowning)
           else:
                print ("No Drowning")
                isDrowning = False
                out = draw_bbox(frame, i, label[j], conf[j],isDrowning)
        j = j+1

    cv2.imshow("Real-time object detection", out)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
