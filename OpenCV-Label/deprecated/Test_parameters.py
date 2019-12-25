import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import argparse
import signal
from functools import wraps
import errno
import os
import copy


image_file = 'GridTest.tif'
image = cv2.imread(image_file)
output = image.copy()

gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
gray = cv2.blur(gray, (3, 3))
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

circles = cv2.HoughCircles(gray,
                           cv2.HOUGH_GRADIENT,
                           dp=3, # resolution of accumulator array.
                           minDist=2,  # number of pixels center of circles should be from each other, hardcode
                           param1=80,
                           param2=66,
                           minRadius=(65 - 3),  # HoughCircles will look for circles at minimum this size
                           maxRadius=(72 + 3)  # HoughCircles will look for circles at maximum this size
                           )

print(circles)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # show the output image
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)

