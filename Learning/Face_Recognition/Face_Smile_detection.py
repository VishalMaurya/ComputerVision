# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:38:33 2020

@author: Vishal
"""

# Importing the libraries
from cv2 import cv2
import os

# os.environ["OPENCV_VIDEOIO_DEBUG"] = '1'

# Loading the cascades
face_cascade = cv2.CascadeClassifier('Face_Recognition/haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Face_Recognition/haarcascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Face_Recognition/haarcascade/haarcascade_smile.xml')

# Detect the face and eye
def detect(gray, frame):
    # get the face from gray image
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        # create the rectangle box on colored image
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        
        # take the boundaries of face in gray and colo image
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # get the face from roi_gray image
        eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 23)
        
        # get smile from roi_gray image
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        
        for (e_x, e_y, e_w, e_h) in eye:
            # create the rectangle box on roi_color image for eye
            cv2.rectangle(roi_color, (e_x,e_y), (e_x+e_w,e_y+e_h), (0,255,0), 2)
            
        for (sx, sy, sw, sh) in smiles: 
            # create the rectangle box on roi_color image for smile
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 255), 2) 
 
    # return frame which is on which both of the boxes are pasted
    return frame

# Connect camera and call face detector
def videoCapture():
    
    # os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = '0'
    # Capture video from 0 --> internal device
    video = cv2.VideoCapture(0)
    while True:
        # get the image from video
        _, frame = video.read()   
        # convert color image to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      
        canvas = detect(gray, frame)
        # render the boxes
        cv2.imshow('Capture', canvas)      
        
        # adding a waiting key till get input 'q' from the user
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release the device
    video.release()
    # destory video captured       
    cv2.destroyAllWindows() 
           
    

# Call videoCapture()
videoCapture()
