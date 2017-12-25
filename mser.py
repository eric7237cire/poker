import cv2
import numpy as np
from configuration import Config as cfg
import os
from PIL import Image
import sys
from card_util import display_image_with_contours

#Create MSER object
mser = cv2.MSER_create()

#Your image path i-e receipt path
file_path = os.path.join(cfg.UNIT_TEST_DATA_DIR, 'bet.png')

image = Image.open(file_path)
image_array = np.array(image)

color_val = 200

mask_white = np.all(image_array >= (color_val,color_val,color_val), axis=-1)
0
image_array[np.logical_not(mask_white)] = [0,0,0]
#mask_black = np.all(image_array <= (25, 25, 25), axis=-1)
#mask = np.logical_not(np.logical_or(mask_white, mask_black))
#image_array[mask] = [50,50,50]

display_image_with_contours(image_array, [])

#sys.exit(0)

#img = cv2.imread(file_path)

# based on https://stackoverflow.com/questions/40078625/opencv-mser-detect-text-areas-python
img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis = img.copy()

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)

cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("text only", text_only)

cv2.waitKey(0)