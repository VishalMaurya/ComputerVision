from cv2 import cv2  
import numpy as np
from helper import stackImages

path = 'resources/cards.jpg'
path2 = 'resources/shapes.png'

def wrap_perspective(path):
    img  = cv2.imread(path)

    w, h = 250, 350
    pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix,(w,h))
    cv2.imshow('Images', img)
    cv2.imshow('Output',imgOutput)
    cv2.waitKey(0)

# wrap_perspective(path)

def join_image(path):
    img  = cv2.imread(path)
    imgh = np.hstack((img, img))
    imgv = np.vstack((img, img))
    cv2.imshow('Horizontal',imgh)
    cv2.imshow('Vertical', imgv)
    cv2.waitKey(0)

# join_image()

def detect_color(path):
    # add track bar
    def empty(x):
        pass

    cv2.namedWindow('TrackBar')
    cv2.resizeWindow('TrackBar', 640, 240)
    cv2.createTrackbar('Hue min', 'TrackBar',0,179,empty)
    cv2.createTrackbar('Hue max', 'TrackBar',179,179,empty)
    cv2.createTrackbar('Saturation min', 'TrackBar',0,255,empty)
    cv2.createTrackbar('Saturation max', 'TrackBar',255,255,empty)
    cv2.createTrackbar('Val min', 'TrackBar',0,255,empty)
    cv2.createTrackbar('Val max', 'TrackBar',255,255,empty)

    while True:
        img  = cv2.imread(path)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # get the values from track bar
        h_min = cv2.getTrackbarPos('Hue min', 'TrackBar')
        h_max = cv2.getTrackbarPos('Hue max', 'TrackBar')
        s_min = cv2.getTrackbarPos('Saturation min', 'TrackBar')
        s_max = cv2.getTrackbarPos('Saturation max', 'TrackBar')
        v_min = cv2.getTrackbarPos('Val min', 'TrackBar')
        v_max = cv2.getTrackbarPos('Val max', 'TrackBar')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max]) 
        mask = cv2.inRange(imgHSV, lower, upper)

        imgMasked = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow('Original image', img)
        # cv2.imshow('HSV image', imgHSV)
        # cv2.imshow('Mask image', mask)
        # cv2.imshow('Masked image', imgMasked)

        imgStack = stackImages(0.6,([img,imgHSV],[mask,imgMasked]))
        cv2.imshow('Stacked Images',imgStack)

        cv2.waitKey(1)

# detect_color(path)

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        cv2.drawContours(imgContour, cnt,-1,(255,0,0),3)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        print(len(approx))  # number of point in shape

        x,y,w,h = cv2.boundingRect(approx)
        if len(approx ) == 3: objType = 'Tri'
        elif len(approx ) == 4:
            aspratio = w/float(h)
            if aspratio>0.95 and aspratio<1.05: objType = 'Square' 
            else: objType = 'Rect'
        elif len(approx ) >4: 
            objType = 'Circle'  
        else: objType = 'Others'
        cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(imgContour, objType,(x+w//2-10,y+h//2-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),2)

def shape_contour(path):
    img = cv2.imread(path)
    imgContour = img.copy()
   
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray scale
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)   # blur image
    imgCanny = cv2.Canny(img, 50, 50) # detecting edges
    imgBlank = np.zeros_like(img)
    getContours(imgCanny, imgContour)
    # cv2.imshow('Original Imge', img)
    # cv2.imshow('Gray Imge', imgGray)
    # cv2.imshow('Blur Imge', imgBlur)

    imgStack = stackImages(0.6,([img, imgGray, imgBlur],[imgCanny, imgContour, imgBlank]))
    cv2.imshow('Stacked Images',imgStack)

    cv2.waitKey(0)

# shape_contour(path2)