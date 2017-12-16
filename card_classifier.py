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
from get_card_suit_and_number import get_suit_and_number
import logging
import sys
logger = logging.getLogger(__name__)


class Card(object):

    def __init__(self, card_index, card_file_name, card_image):
        self.card_index = card_index
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

def diff_images(card1, card2):

    return diff_2d_array(card1.suit_image,
                         card2.suit_image)
    + diff_2d_array(card1.number_image, card2.number_image)



def main():
    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images(cfg.CARD_DATA_PATH))

    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    rawImages = []
    features = []
    labels = []
    cards = []

    test_cards = []
    cards_to_eval = []

    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']
    suits = ['d', 'c', 's', 'h']

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        image = Image.open(imagePath)

        file_name = os.path.basename(imagePath)

        # File name should be like ac for Ace of clubs or 2d for 2 of diamonds
        label = ranks.index(file_name[0]) + 13 * suits.index(file_name[1])

        logger.info("Label is {} for file name {}".format(label, file_name))

        card = Card(card_index=label, card_file_name=file_name, card_image=image)

        if file_name[2] == '_':
            cards_to_eval.append(card)
        else:
            test_cards.append(card)

        card.suit_image, card.number_image = get_suit_and_number(image)

        cards.append(card)

        # update the raw images, features, and labels matricies,
        # respectively
        rawImages.append(image)

        labels.append(label)

        # show an update every 1,000 images
        # if i > 0 and i % 1000 == 0:
        logger.info("[INFO] processed {}/{}".format(i, len(imagePaths)))

    for card in cards_to_eval:
        logger.debug(f"Evaluating {card.card_file_name} / {card.card_index} ")

        card_diffs = [diff_images(card, t) for t in test_cards]

        index = np.argmin(card_diffs, axis=0)

        print(f"Closest to {test_cards[index].card_file_name}")

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
