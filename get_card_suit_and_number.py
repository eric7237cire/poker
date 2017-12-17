import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from skimage import measure
import os
import cv2
from configuration import Config as cfg
from PIL import Image
import numpy as np
from card_util import *
from PIL import Image, ImageDraw
import logging

from scipy.misc import imresize
from matplotlib.path import Path

logger = logging.getLogger(__name__)


def main():
    file_path = os.path.join(
        cfg.CARD_DATA_PATH,
        '..',
        'unit_test_data',
        'screenshot_with_hole_cards.png'
        #'7s_2.png'
        # '2h.png'
        # '9s.png'
    )

    file_path = os.path.join(
        cfg.CARD_DATA_PATH,

        '8h_2.png'
        # '2h.png'
        # '9s.png'
    )

    # image = cv2.imread(file_path)
    image = Image.open(file_path)

    image_array = np.array(image)

    #image_array = cfg.PLAYER_DIMENSIONS[0].clip_2d_array(image_array)

    cropped_image = Image.fromarray(image_array)


    card_contours = get_suit_and_number(cropped_image, has_2_cards=False)

    display_image_with_contours(card_contours[0].grey_array,
                                [x.contour for x in card_contours])
    #display_image_with_contours(image_array, [])


def main2():
    file_path = os.path.join(
        cfg.CARD_DATA_PATH,
        '7s_2.png'
        # '2h.png'
        #'9s.png'
    )

    # image = cv2.imread(file_path)
    image = Image.open(file_path)




class CardContour():

    def __init__(self):
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None

        # List of points y,x
        self.contour = None
        self.grey_array = None


def get_suit_and_number(image, has_2_cards=False):
    """
    Takes a PIL.Image
    :param image:
    :return: Suit card contour object and number card contour object
    """
    grey_array = card_to_grayscale_2d_array(image)

    card_contours = []

    for idx, contour in enumerate(find_contours_in_card(
            image, grey_array=grey_array,
            min_width=5,
            max_width=15)):

        card_contour = CardContour()
        card_contour.min_y, card_contour.min_x = np.min(contour, axis=0)
        card_contour.max_y, card_contour.max_x = np.max(contour, axis=0)
        card_contour.contour = contour
        card_contour.grey_array = grey_array
        card_contours.append(card_contour)



    sorted_by_x = sorted(card_contours, key=lambda c: c.min_x)

    if has_2_cards:


        sorted_contours_1 = sorted(sorted_by_x[0:2], key=lambda c: c.min_y)
        sorted_contours_2 = sorted(sorted_by_x[-3:-1], key=lambda c: c.min_y)

        # suit, number
        return sorted_contours_1[1], sorted_contours_1[0], sorted_contours_2[1], sorted_contours_2[0]
    else:
        sorted_by_y = sorted(sorted_by_x[0:2], key=lambda c: c.min_y)
        return sorted_by_y[1], sorted_by_y[0]


if __name__ == "__main__":
    main()
