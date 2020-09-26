from cv2 import cv2
import numpy as np
def wrap_perspective(img,edges):
    width, height = 250,350
    pts1 = np.float32([edges[0],edges[1],edges[2],edges[3]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(width,height))
    cv2.imshow("Output Image ", imgOutput)



def mousePoints(event,x,y,flags,params):
    global counter
    global edges
    if event == cv2.EVENT_LBUTTONDOWN:
        edges[counter] = x,y
        counter = counter + 1
        print(edges)

def reorder (myPoints):
    # myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)

    add = myPoints.sum(1)
    diff = np.diff(myPoints,axis=1)
    
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def DetectClickandCrop(path):
    global edges
    edges = np.zeros((4,2),np.int)
    global counter 
    counter = 0

    img = cv2.imread(path)
    
    while True:
        if counter == 4:
            mypoint = reorder(edges)
            wrap_perspective(img, mypoint)
            counter = 0

        for x in range (0,4):
            cv2.circle(img,(edges[x][0],edges[x][1]),3,(0,255,0),cv2.FILLED)

        cv2.imshow("Original Image ", img)
        cv2.setMouseCallback("Original Image ", mousePoints)
        cv2.waitKey(1)

path = './resources/cards.jpg'

DetectClickandCrop(path)