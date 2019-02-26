
from train import *
import os 

print cv2.__version__
path = os.path.dirname(__file__)
if len(path) != 0:
    os.chdir(os.path.dirname(path))

try:
    os.mkdir('faces')
    os.mkdir('faces/users')
except:
    print 'err'
    pass
capture = raw_input('Capture image for training[y/n]: ')
if str(capture).lower() == 'y':
    cap = cv2.VideoCapture(0)
    count=0
    while True:

        _,frame = cap.read()
        detectedFaces = extractFace(frame)

        if detectedFaces is None:
            cv2.putText(frame,'Face Not Detected',(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.imshow('Face Selector',frame)
        else:
            count+=1
            face = cv2.resize(detectedFaces,(200,200))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

            file_name_path = './faces/users/'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(detectedFaces,str(count),(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
            cv2.imshow('Face Selector',detectedFaces)

        k =cv2.waitKey(1) 


        if k==27 or count == 300 :
            break

    cv2.destroyAllWindows()
    cap.release()

##########################
##### TRAINING MODEL #####
##########################

data_path = './faces/users/'

files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,f))]

trainingData,Labels = [],[]


for i,f in enumerate(files):
    image_path = data_path + files[i]
    images = cv2.imread(image_path,0)
    #print type(images)
    trainingData.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels =np.asarray(Labels,dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(trainingData),np.asarray(Labels))


############################
###Run Facial Recognition###
############################
cap = cv2.VideoCapture(0)

while True:
    
    ret,frame = cap.read()

    try:
        face = extractFace(frame)
        face = cv2.resize(face,(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        results = model.predict(face)

        if results[1] < 500:
            confidence = int(100*(1-results[1]/300))
            dis_str = str(confidence)+' confident'
        
        if confidence > 87:
            cv2.putText(frame,dis_str,(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            cv2.imshow('Detection',frame)
        else:
            cv2.putText(frame,'Not Detected',(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            cv2.imshow('Detection',frame) 
    except:
        cv2.putText(frame,'No Face',(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imshow('Detection',frame)

    
    k = cv2.waitKey(1)
    if k==27:
        break

cap.release()
cap.destroyAllWindows()   
