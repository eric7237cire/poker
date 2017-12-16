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
        '2h.png'
    )

    # image = cv2.imread(file_path)
    image = Image.open(file_path)


def get_suit_and_number(image):
    grey_array = card_to_grayscale_2d_array(image)

    contours = []

    points = generate_points_list(cfg.CARD_WIDTH_PIXELS, cfg.CARD_HEIGHT_PIXELS)

    number_image = None
    suit_image = None

    for idx, contour in enumerate(find_contours_in_card(image, grey_array=grey_array)):

        boolean_mask = extract_polygon_mask_from_contour(
            contour,
            height=cfg.CARD_HEIGHT_PIXELS, width=cfg.CARD_WIDTH_PIXELS,
            all_grid_points_list=points
        )

        extracted_image = extract_image_with_mask(
            grey_array, boolean_mask, background_color=255)

        # show_image_and_contour(extracted_image, contour)

        # print(f"Found contour {min_x} {max_x} {min_y} {max_y}\n{contour}")

        if idx == 0:
            number_image = extracted_image
        elif idx == 0:
            suit_image = extracted_image
        else:
            logging.warning("Extracted more than one image")

        contours.append(contour)

    return suit_image, number_image


if __name__ == "__main__":
    main()
