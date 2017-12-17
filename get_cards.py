# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import shutil
import imutils
import cv2
import sys
import os
from datetime import datetime
from card_util import *

from configuration import Config as cfg
from get_screenshot import capture_screenshot
from card_classifier import *

from PIL import Image, ImageDraw

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



def main():

    card_classifier = CardClassifier()

    try:
        os.makedirs(cfg.SCREENSHOTS_PATH, exist_ok=True)
        shutil.rmtree(cfg.EXTRACTED_IMAGES_PATH, ignore_errors=True)
        os.makedirs(cfg.EXTRACTED_IMAGES_PATH, exist_ok=True)
    except Exception as ex:
        print(ex)

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d__%H_%M_%S_%f")

    file_path = os.path.join(cfg.SCREENSHOTS_PATH, 'screenshot_{}.png'.format(formatted_time))
    capture_screenshot("chrome", output_file_path=file_path)

    #file_path=os.path.join(cfg.SCREENSHOTS_PATH, "screenshot_2017_12_15__19_59_16_686117.png")

    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(file_path)

    bw = get_black_and_white_image(image)
    cnts = find_contours(bw)

    image_copy = image.copy()

    image_array = np.array(image)

    # loop over the contours individually
    for idx, contour in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(contour) < 10:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < cfg.CARD_WIDTH_PIXELS - 5 or w > cfg.CARD_WIDTH_PIXELS + 5:
            continue

        if h < cfg.CARD_HEIGHT_PIXELS -5 or h > cfg.CARD_HEIGHT_PIXELS + 5:
            continue

        crop_img = image_array[y:y + h + 1, x:x + w + 1]

        clip_and_save(
            p_orig_image=image_copy,
            x=x,
            y=y,
            w=cfg.CARD_WIDTH_PIXELS,
            h=cfg.CARD_HEIGHT_PIXELS,
            file_name=f"sub_image_{idx:04}.png"
        )

        card_image = Image.fromarray(crop_img)

        c = card_classifier.evaluate_card(card_image)

        print(f"Classified {idx} as {c}")





    # show the output image
    #cv2.imshow("Image", bw)
    #cv2.imshow("Image", image_copy)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
