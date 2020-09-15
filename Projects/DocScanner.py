from cv2 import cv2
import numpy as np

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
        

        cv2.imshow('Output', imgFinal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break