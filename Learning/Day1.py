from cv2 import cv2
import numpy as np

path = 'resources/vishal.jpg'

def read_image(path):
    img = cv2.imread(str(path))
    cv2.imshow('Output', img)
    cv2.waitKey(0)

#read_image(path)

def read_video(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        success, img = cap.read()
        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#read_video('resources/Netflix.mp4')

def read_cam(camId):
    cap = cv2.VideoCapture(camId)
    cap.set(3, 1080) # setting width 
    cap.set(4, 640) # setting height
    cap.set(10, 1000) # set brightness

    while True:
        success, img = cap.read()
        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

#read_cam(0)

def resize_crop_img(path):
    img = cv2.imread(path)
    imgResize = cv2.resize(img, (500, 600))
    imgCrop = imgResize[:300,]

    cv2.imshow('Resize image', imgResize)
    cv2.imshow('Crop image', imgCrop)

    cv2.waitKey(0)

#resize_crop_img(path)

def exp_with_img(path):
    img = cv2.imread(str(path))
    kernel = np.ones((5,5), np.uint8)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray scale
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)   # blur image

    imgCanny = cv2.Canny(img, 150, 200) # detecting edges
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)   # increase the thickness of edge detected
    imgErode = cv2.erode(imgDialation,kernel,iterations = 1)    # decrease the thickness of edge detected

    cv2.imshow('Gray Imge', imgGray)
    cv2.imshow('Blur Imge', imgBlur)
    cv2.imshow('Canny Imge', imgCanny)
    cv2.imshow('Dialation Imge', imgDialation)
    cv2.imshow('Erode Imge', imgErode)


    cv2.waitKey(0)

#exp_with_img(path)

def img_drawing():
    img = np.zeros((512,512,3), np.uint8)
#    img[:] = 0,0,255
    cv2.line(
        img,
        (0,0),
        (img.shape[1],img.shape[0]),
        color=(0,255,0),
        thickness=2
    )
    cv2.rectangle(
        img,
        (0,0),
        (150,150),
        color=(0,55,100),
        thickness=2
    )
    cv2.circle(
        img,
        (100,200),
        radius=10,
        color=(255,250,0),
        thickness=2
    )
    cv2.putText(
        img,
        "OpenCv",
        (300,300),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1,
        color=(0,150,0),
        thickness=2
    )

    cv2.imshow('Draw image', img)
    cv2.waitKey(0)

#img_drawing()

