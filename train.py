
import cv2 
import numpy as np 

def extractFace(img):
    face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grayImg,1.3,5)

    if faces is ():
        return None 

    for(x,y,w,h) in faces:
        cropped_faces = img[y:y+h,x:x+w]

    return cropped_faces