
from train import *
import os 

path = os.path.dirname(__file__)
if len(path) != 0:
    os.chdir(os.path.dirname(path))

try:
    os.mkdir('faces')
    os.mkdir('faces/users')
except:
    print 'err'
    pass

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


    if k==27 or count == 100 :
        break

cv2.destroyAllWindows()
cap.release()

##########################
##### TRAINING MODEL #####
##########################

data_path = './faces/users/'

files = [f for f in os.listdir(data_path) if isfile(join(data_path,f))]

trainingData,Labels = [],[]


for i,files in enumerate(files):
    image_path = data_path + files[i]
    images = cv2.imread(image_path,0)
    trainingData.append(np.asarray(images,dtype=uint8))
    Labels.append(i)

Labels =np.asarray(Labels,dtype=np.int32)

model = cv2.face.createLBPHFaceRecognizer()
