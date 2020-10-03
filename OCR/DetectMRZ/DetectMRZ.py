# import necessary packages
from imutils import paths
from cv2 import cv2
import numpy as np
import argparse
import imutils

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image directory")
args = vars(ap.parse_args())

# initialize rectangle and square kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
squrKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))


# loop over the image directory
for imagePath in paths.list_images(args["image"]):
    # load image file
    image = cv2.imread(imagePath)
    image = imutils.resize(image, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth the image using 3x3 kernel, then apply blackhat morphological operation
    # to find the dark regions on the light background
    gray = cv2.GaussianBlur(gray, (3,3) ,0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # compute the gradient intensity of blackhat image using Sobel algorithm
    # and scale into range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # apply the closing operation using the rectangular kernel to close gaps
    # in between letters -- then apply the Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0 , 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # apply the close operation this time using squrKernel to close the gaps
    # in between lines of MRZ, then perform erosion to break apart connect components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, squrKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    reducePercentage = int(image.shape[1] * 0.05)
    thresh[:, 0:reducePercentage] = 0
    thresh[:, image.shape[1] - reducePercentage:] = 0

    # find contours in image ans sort them by thier size
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    ROI = []
    for contour in contours:
        # compute bounding box, aspratio, width to the width of the image
        (x, y, w, h) = cv2.boundingRect(contour)
        aspratio = w / float(h)
        crWidth = w / float(gray.shape[1])
        print(aspratio, crWidth)
        # check the acceptance criteria
        if aspratio > 5 and crWidth > 0.7 :
            # apply padding to re-grow image since used erosions
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x -pX, y-pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))
            print(aspratio)
            # get the ROI and draw bounding box
            ROI = image[y:y+h, x:x+w].copy()
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 255), 2)
            cv2.imshow("ROI Image",ROI)

            break

    cv2.imshow("Input Image",image)
    cv2.imshow("test", thresh)
    cv2.waitKey(0)

    
