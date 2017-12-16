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

class Card(object):

    def __init__(self, card_index, card_file_name, card_image):
        self.card_index = card_index
        self.card_file_name = card_file_name
        self.card_image = card_image

        self.suit_image=None
        self.number_image = None


def diff_images(card1, card2):
    # cv2.imshow("Image1", img1)
    # cv2.imshow("Image2", img2)

    # cv2.imshow("Image", image_copy)

    diff = np.absolute(card1.suit_image, card2.suit_image)

    # cv2.imshow("P Image1", p_img1)
    # cv2.imshow("P Image2", p_img2)
    # cv2.imshow("diff", diff)

    # diff = cv2.GaussianBlur(diff,(5,5),5)
    # flag, diff = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
    diff_sum = np.sum(diff)

    # cv2.imshow("diff2", diff)
    # print (diff_sum)
    # cv2.waitKey(0)

    return diff_sum


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

        print("Label is {} for file name {}".format(label, file_name))

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
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    for card in cards_to_eval:
        print(f"Evaluating {card.card_file_name} / {card.card_index} ")

        card_diffs = [diff_images(card, t) for t in test_cards]

        index = np.argmin(card_diffs, axis=0)

        print(f"Closest to {test_cards[index].card_file_name}")

    return



if __name__ == "__main__":
    main()
