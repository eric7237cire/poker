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
from card_util import *

logger = logging.getLogger(__name__)


def main():
    file_path = os.path.join(
        cfg.CARD_DATA_PATH,
        '..',
        'unit_test_data',
        'screenshot_with_hole_cards.png'
        # '7s_2.png'
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

    # image_array = cfg.PLAYER_DIMENSIONS[0].clip_2d_array(image_array)

    cropped_image = Image.fromarray(image_array)

    card_contours = get_suit_and_number(cropped_image, has_2_cards=False)

    display_image_with_contours(card_contours[0].grey_array,
                                [x.contour for x in card_contours])
    # display_image_with_contours(image_array, [])


def main2():
    file_path = os.path.join(
        cfg.CARD_DATA_PATH,
        '7s_2.png'
        # '2h.png'
        # '9s.png'
    )

    # image = cv2.imread(file_path)
    image = Image.open(file_path)


def get_suit_and_number(image, has_2_cards=False):
    """
    Takes a PIL.Image
    :param image:
    :return: Suit card contour object and number card contour object
    """
    grey_array = card_to_grayscale_2d_array(image)

    if not has_2_cards:
        # chips can cover up bottom
        grey_array = grey_array[0:28, :]

    card_contours = []

    for idx, contour in enumerate(find_contours(
            grey_array=grey_array,
            min_width=5,
            max_width=15)):
        card_contours.append(contour)

    if has_2_cards:
        #display_image_with_contours(grey_array, [c.points_array for c in card_contours])
        pass

    sorted_by_x = sorted(card_contours, key=lambda c: c.bounding_box.min_x)

    if has_2_cards:

        if len(sorted_by_x) < 4:
            logger.warning("Not enough images for hole cards")
            # display_image_with_contours(grey_array, [c.contour for c in card_contours])
            return None, None, None, None

        sorted_contours_1 = sorted(sorted_by_x[0:2], key=lambda c: c.bounding_box.min_y)
        sorted_contours_2 = sorted(sorted_by_x[-3:-1], key=lambda c: c.bounding_box.min_y)

        display_image_with_contours(grey_array, [c.points_array for c in sorted_by_x[3:5]])

        # suit, number
        return sorted_contours_1[1], sorted_contours_1[0], sorted_contours_2[1], sorted_contours_2[0]
    else:
        if len(sorted_by_x) < 2:
            logger.warning("Not enough images for cards")
            display_image_with_contours(grey_array, [c.contour for c in card_contours])
            return None, None

        sorted_by_y = sorted(sorted_by_x[0:2], key=lambda c: c.bounding_box.min_y)
        return sorted_by_y[1], sorted_by_y[0]


if __name__ == "__main__":
    main()
