from cv2 import cv2
import numpy as np
import random


def wrap_perspective(img,edges):
    width, height = 250,350
    pts1 = np.float32([edges[0],edges[1],edges[2],edges[3]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(width,height))
    cv2.imshow("Output Image ", imgOutput)

    # if cv2.waitKey(1) & 0xFF == ord('s'):
    #     cv2.imwrite("Projects/resources/scanned/Cropped_"+str(random.randint(1, 99))+".jpg",imgOutput)
    #     cv2.rectangle(imgOutput,(0,200),(640,300),(0,255,0),cv2.FILLED)

    #     cv2.putText(imgOutput,"Scan Saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,
    #                     2,(0,0,255),2)
    #     cv2.imshow("Result",imgOutput)
    #     cv2.waitKey(50)



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
            img = cv2.imread(path)

        for x in range (0,4):
            cv2.circle(img,(edges[x][0],edges[x][1]),3,(0,255,0),cv2.FILLED)     

        cv2.imshow("Original Image ", img)
        cv2.setMouseCallback("Original Image ", mousePoints)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




path = './resources/cards.jpg'

DetectClickandCrop(path)