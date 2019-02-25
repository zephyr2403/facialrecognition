
from train import *

cap = cv2.VideoCapture(0)
trainX = raw_input('Do You want to train(y/n):')
if str(trainX).lower()=='y':
   
    trainN = raw_input('Press "Y" to train on new data, anything else to add new Face: ')
    if str(trainN).lower() == 'y':
        while True:

            _,frame = cap.read()
            detectedFaces = extractFace(frame)

            k =cv2.waitKey(1)

            if detectedFaces is  None:
                continue

            cv2.imshow('Detected',detectedFaces)
            if k==27:
                break

        cv2.destroyAllWindows()
        cap.release()