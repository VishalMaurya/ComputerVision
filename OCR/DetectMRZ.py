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
squrKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

while True:
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
    
    cv2.imshow("Op",thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
