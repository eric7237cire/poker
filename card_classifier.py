# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from configuration import Config as cfg
from PIL import Image, ImageDraw
from shapely.affinity import translate
from get_card_suit_and_number import get_suit_and_number
import logging
import sys
from shapely.geometry import Polygon
logger = logging.getLogger(__name__)

from card_util import *

class Card(object):

    def __init__(self, card_index, card_file_name, card_image):
        self.card_id = card_index
        self.card_file_name = card_file_name
        self.card_image = card_image

        self.suit_image = None
        self.number_image = None


def diff_2d_array(image1, image2):
    if image1 is None or image2 is None:
        return 10000000000000
    # See also https://stackoverflow.com/questions/35777830/fast-absolute-difference-of-two-uint8-arrays
    return np.sum(np.absolute(np.int16(image1) -
                              np.int16(image2)))


def diff_polygons(contour1, contour2):

    if contour1 is None or contour2 is None:
        return 10000000000000

    poly1 = Polygon(get_contour_xy(contour1))
    poly2 = Polygon(get_contour_xy(contour2))

    poly1 = translate(poly1, xoff=-poly1.bounds[0], yoff=-poly1.bounds[1])
    poly2 = translate(poly2, xoff=-poly2.bounds[0], yoff=-poly2.bounds[1])

    intersecting_area = poly1.intersection(poly2).area

    return poly1.area + poly2.area - 2 * intersecting_area


def diff_images(card1, card2):
    return diff_polygons(card1.suit_image,
                         card2.suit_image) + \
           diff_polygons(card1.number_image, card2.number_image)

class CardClassifier():
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']
    SUITS = ['d', 'c', 's', 'h']

    def get_card_id(self,rank, suit):
        # File name should be like ac for Ace of clubs or 2d for 2 of diamonds
        return self.RANKS.index(rank) + 13 * self.SUITS.index(suit)

    def get_card_string(self, card_id):

        suit = int(card_id / 13)
        rank = card_id % 13

        return "{} of {}".format(self.RANKS[rank], self.SUITS[suit])

    def __init__(self):
        imagePaths = list(paths.list_images(cfg.CARD_DATA_PATH))

        # initialize the raw pixel intensities matrix, the features matrix,
        # and labels list
        rawImages = []
        file_name_to_card = {}
        features = []
        labels = []
        cards = []

        self.test_cards = []
        self.cards_to_eval = []



        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            image = Image.open(imagePath)

            file_name = os.path.basename(imagePath)

            # File name should be like ac for Ace of clubs or 2d for 2 of diamonds
            label = self.get_card_id(rank=file_name[0], suit=file_name[1])

            logger.info("Label is {} for file name {}".format(label, file_name))

            card = Card(card_index=label, card_file_name=file_name, card_image=image)

            if file_name[2] == '_':
                self.cards_to_eval.append(card)
            else:
                self.test_cards.append(card)

            card.suit_image, card.number_image = get_suit_and_number(image)

            cards.append(card)

            file_name_to_card[file_name] = card

            # update the raw images, features, and labels matricies,
            # respectively
            rawImages.append(image)

            labels.append(label)

            # show an update every 1,000 images
            # if i > 0 and i % 1000 == 0:
            logger.info("[INFO] processed {}/{}".format(i, len(imagePaths)))

    def evaluate_card(self, card_image):
        card = Card(card_index=None, card_file_name=None, card_image=card_image)
        card.suit_image, card.number_image = get_suit_and_number(card_image)

        if card.number_image is None:
            return None

        card_diffs = [diff_images(card, t) for t in self.test_cards]

        # index = np.argmin(card_diffs, axis=0)
        index = np.argmin(card_diffs, axis=0)

        return self.test_cards[index].card_id


    def evaluate_accuracy(self):
        for card in self.cards_to_eval:
            logger.debug(f"Evaluating {card.card_file_name} / {card.card_index} ")

            card_diffs = [diff_images(card, t) for t in self.test_cards]

            # index = np.argmin(card_diffs, axis=0)
            index = np.argmin(card_diffs, axis=0)

            print(f"Closest to {test_cards[index].card_file_name}")

def main():
    # grab the list of images that we'll be describing
    print("[INFO] describing images...")


    c1 = file_name_to_card["2h.png"]
    c2 = file_name_to_card["2h_2.png"]
    c3 = file_name_to_card["5h.png"]

    #display_image_with_contours(c1.card_image, [
        #c1.number_image,
        #c2.number_image,
        #c3.number_image
    #])

    #d1 = diff_images(c1, c2)
    #d2 = diff_images(c3, c2)

    #print(d1)
    #print(d2)
    #return



    return


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

    main()
