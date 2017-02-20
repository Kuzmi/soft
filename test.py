import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path
from skimage.morphology import dilation, erosion, square, disk, diamond
from skimage.measure import label,regionprops
from vector import distance, pnt2line
from sklearn.datasets import fetch_mldata
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.models import model_from_json



id = -1 
def tranAndsave():
    mnist = fetch_mldata('MNIST original')
    data   = mnist.data / 255.0
    labels = mnist.target.astype('int')

    train_rank = 10000
    train_subset = np.random.choice(data.shape[0], train_rank)

    train_data = data[train_subset]
    train_labels = labels[train_subset]

    train_out = to_categorical(train_labels, 10)


    model = Sequential()
    model.add(Dense(70, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('relu'))

    sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)



    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Model sacuvan")
    return model

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

def loadModel():
    json_file = open('model.json', 'r')
    jsonModel = json_file.read()
    json_file.close()
    model = model_from_json(jsonModel)
    model.load_weights("model.h5")
    print("Model ucitan")
    return model


def getNumPic(element,image):
    retSize = (28,28)
    retCenter = element['center']
    retLoc = (retCenter[0]-retSize[0]/2, retCenter[1]-retSize[1]/2)
    tempIm = image[retLoc[0]:retLoc[0]+retSize[0],retLoc[1]:retLoc[1]+retSize[1]]
    retImage = tempIm.reshape(784)
    retImage = retImage/255.
    return retImage







def getGray(image):
    imageGray = np.ndarray((image.shape[0], image.shape[1]))
    for i in np.arange(0, image.shape[0]):
        for j in np.arange(0, image.shape[1]):
            if image[i, j, 0] > 240 and image[i, j, 1] < 140 and image[i, j, 2] < 140:
                imageGray[i, j] = 255
            else:
                imageGray[i, j] = 0
    imageGray = imageGray.astype('uint8')
    return imageGray

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

def nextId():
    global id
    id += 1
    return id




def main():

    if(os.path.exists('model.json')):
        model = loadModel()
    else:
       model= tranAndsave()


    cap = cv2.VideoCapture("Videos/video-0.avi")
    frameCount = 0
    begLinePoint = []
    endLinePoint=[]
    suma=0
    kernel = np.ones((2,2),np.uint8)
    elements=[]

    while(cap.isOpened()):
        ret, frame = cap.read()
        if(frameCount == 0):
            lineImg = getGray(frame)
            lineImgTh = lineImg > 150
            lineImgDil = dilation(lineImgTh, selem=diamond(3))
            lineLabel = label(lineImgDil)
            lineRegions = regionprops(lineLabel)
            begLinePoint = [lineRegions[0].bbox[3],lineRegions[0].bbox[0]]
            endLinePoint = [lineRegions[0].bbox[1],lineRegions[0].bbox[2]]
        lower = np.array([230, 230, 230],dtype = "uint8")
        upper = np.array([255 , 255 , 255],dtype = "uint8")
        mask = cv2.inRange(frame, lower, upper)   
        imgBase = 1.0*mask

        imgBase = cv2.dilate(imgBase,kernel)
        imgBase=cv2.erode(imgBase,kernel)
        imgBase = cv2.dilate(imgBase,kernel)  

        baseLabel = label(imgBase)
        regions = regionprops(baseLabel)

        for region in regions:
            number = {'center' : region.centroid,  'frame' : frameCount}
            res = inRange(20,number,elements)

            if len(res) == 0:
                number['id'] = nextId()
                number['pass'] = False
                elements.append(number)
            elif len(res) == 1:
                res[0]['center'] = number['center']
                res[0]['frame'] = frameCount

        for elem in elements:
            t = frameCount - elem['frame'] 
            if(t<3):
                dist, pnt, r = pnt2line(elem['center'], begLinePoint, endLinePoint)
                if(r>0):
                    if(dist<6):
                        if elem['pass'] == False:
                            elem['pass'] = True
                            fImage = getNumPic(elem,imgBase)
                            tt = model.predict(np.array([fImage]), verbose=1)
                            tMax = tt.argmax(axis=1)
                            suma+= tMax[0]
                            print suma
        frameCount+=1
         
           
    cap.release()
    cv2.destroyAllWindows()


main()