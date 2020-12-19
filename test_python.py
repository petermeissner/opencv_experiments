import cv2 as cv
import sys


img = cv.imread(cv.samples.findFile("2020-05-02 10.18.57.jpg"))



if img is None:
    sys.exit("Could not read the image.")


cv.imwrite("starry_night.png", img)
