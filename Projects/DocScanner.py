from cv2 import cv2
import numpy as np
from helper import stackImages


def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

def getContours(img, imgContour):
    biggest = np.array([])
    maxArea = 0
    while True:
        if img.shape[1] == 0:
            continue
        contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>5000:
                #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                if area >maxArea and len(approx) == 4:
                    biggest = approx
                    maxArea = area
        if contours is not None:
            cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
        return biggest

def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)

    add = myPoints.sum(1)
    diff = np.diff(myPoints,axis=1)
    
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped

def DocScanner():
    global widthImg 
    global heightImg 
    widthImg = 540
    heightImg = 640
    cap = cv2.VideoCapture(0)
    cap.set(3, widthImg) # setting width 
    cap.set(4, heightImg) # setting height
    cap.set(10, 150) # set brightness
    while True:
        _ , img = cap.read()
        if img is None:
            continue
        cv2.resize(img, (widthImg, heightImg))
        imgContour = img.copy()
        imgThres = preProcessing(img)
        biggest = getContours(imgThres, imgContour)

        if biggest.size !=0:
            imgWarped=getWarp(img,biggest)
            # imageArray = ([img,imgThres],
            #           [imgContour,imgWarped])
            imageArray = ([imgContour, imgWarped])
            cv2.imshow("ImageWarped", imgWarped)
        else:
            # imageArray = ([img, imgThres],
            #               [img, img])
            imageArray = ([imgContour, img])

        stackedImages = stackImages(0.6,imageArray)
        cv2.imshow("WorkFlow", stackedImages)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


DocScanner()