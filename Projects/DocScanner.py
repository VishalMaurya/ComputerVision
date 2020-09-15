from cv2 import cv2
import numpy as np


def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

def DocScanner():
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
        imgCount = img.copy()


        cv2.imshow('Output', imgCount)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break