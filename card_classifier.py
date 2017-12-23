# import the necessary packages
from imutils import paths
import logging

from get_card_suit_and_number import get_suit_and_number
from configuration import Config as cfg
import numpy as np
from card_util import diff_polygons, display_image_with_contours
from PIL import Image
import os

logger = logging.getLogger(__name__)
trace_logger = logging.getLogger(__name__ + "_trace")

class Card(object):

    def __init__(self, card_index, card_file_name, card_image):
        self.card_id = card_index
        self.card_file_name = card_file_name
        self.card_image = card_image

        # Contour objects
        self.suit_image = None
        self.number_image = None


def diff_2d_array(image1, image2):
    if image1 is None or image2 is None:
        return 10000000000000
    # See also https://stackoverflow.com/questions/35777830/fast-absolute-difference-of-two-uint8-arrays
    return np.sum(np.absolute(np.int16(image1) -
                              np.int16(image2)))


def diff_images(card1, card2, scale_polygons=True):
    return diff_polygons(card1.suit_image,
                         card2.suit_image, scale_polygons=scale_polygons) + \
           diff_polygons(card1.number_image, card2.number_image, scale_polygons=scale_polygons)


class CardClassifier(object):
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']
    SUITS = ['d', 'c', 's', 'h']

    SUITS_VERBOSE = ['Diamonds', 'Clubs', 'Spades', 'Hearts']

    def get_card_id(self, rank, suit):
        # File name should be like ac for Ace of clubs or 2d for 2 of diamonds
        return self.RANKS.index(rank) + 13 * self.SUITS.index(suit)

    def get_card_string(self, card_id):

        if not isinstance(card_id, int):
            return None

        suit = int(card_id / 13)
        rank = card_id % 13

        return "{} of {}".format(self.RANKS[rank].upper(), self.SUITS_VERBOSE[suit])

    def get_card_short_string(self, card_id):

        if not isinstance(card_id, int):
            return None

        suit = int(card_id / 13)
        rank = card_id % 13

        return "{}{}".format(self.RANKS[rank].upper(), self.SUITS[suit])

    def get_test_card(self, rank, suit):

        card_id = self.get_card_id(rank, suit)

        for card in self.test_cards:
            if card.card_id == card_id:
                return card

        return None

    def __init__(self):
        image_paths = list(paths.list_images(cfg.CARD_DATA_PATH))

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
        for (i, imagePath) in enumerate(image_paths):
            image = Image.open(imagePath)

            file_name = os.path.basename(imagePath)

            # File name should be like ac for Ace of clubs or 2d for 2 of diamonds
            label = self.get_card_id(rank=file_name[0], suit=file_name[1])

            trace_logger.debug("Label is {} for file name {}".format(label, file_name))

            card = Card(card_index=label, card_file_name=file_name, card_image=image)

            if file_name[2] == '_':
                self.cards_to_eval.append(card)
            else:
                self.test_cards.append(card)

            card.suit_image, card.number_image = get_suit_and_number(image, clip_bottom=False)

            cards.append(card)

            file_name_to_card[file_name] = card

            # update the raw images, features, and labels matricies,
            # respectively
            rawImages.append(image)

            labels.append(label)

            # show an update every 1,000 images
            # if i > 0 and i % 1000 == 0:
            trace_logger.debug("[INFO] processed {}/{}".format(i, len(image_paths)))

    def evaluate_hole_card_image(self, hole_card_image):

        card1 = Card(card_index=None, card_file_name=None, card_image=hole_card_image)
        card2 = Card(card_index=None, card_file_name=None, card_image=hole_card_image)

        card1.suit_image, card1.number_image, card2.suit_image, card2.number_image = \
            get_suit_and_number(hole_card_image, has_2_cards=True)

        if False:
            # Use to debug hole card detection
            c1 = self.get_test_card('9', 's')
            c2 = self.get_test_card('9', 'd')

            display_image_with_contours(np.array(hole_card_image), [
                card1.number_image.points_array, card1.suit_image.points_array,
                c1.number_image.points_array, c2.suit_image.points_array
            ])

        return self.evaluate_suit_and_number_images(card1, scale_polygons=True), self.evaluate_suit_and_number_images(card2, scale_polygons=True)

    def evaluate_card(self, card_image, display=False):
        card = Card(card_index=None, card_file_name=None, card_image=card_image)
        card.suit_image, card.number_image = get_suit_and_number(card_image)

        if display:
            display_image_with_contours(np.array(card_image), [card.suit_image.points_array,
                                                                     card.number_image.points_array])

        return self.evaluate_suit_and_number_images(card)

    def evaluate_suit_and_number_images(self, card, scale_polygons=False):

        if card.number_image is None:
            return None

        card_diffs = [diff_images(card, t, scale_polygons=scale_polygons) for t in self.test_cards]

        # index = np.argmin(card_diffs, axis=0)
        index = np.argmin(card_diffs, axis=0)

        return self.test_cards[index].card_id

    def evaluate_accuracy(self):
        for card in self.cards_to_eval:
            logger.debug(f"Evaluating {card.card_file_name} / {card.card_id} ")

            card_diffs = [diff_images(card, t) for t in self.test_cards]

            # index = np.argmin(card_diffs, axis=0)
            index = np.argmin(card_diffs, axis=0)

            print(f"Closest to {self.test_cards[index].card_file_name}")


def main():
    # grab the list of images that we'll be describing
    print("[INFO] describing images...")

    classifier = CardClassifier()

    classifier.evaluate_accuracy()

    # c1 = file_name_to_card["2h.png"]
    # c2 = file_name_to_card["2h_2.png"]
    # c3 = file_name_to_card["5h.png"]

    # display_image_with_contours(c1.card_image, [
    # c1.number_image,
    # c2.number_image,
    # c3.number_image
    # ])

    # d1 = diff_images(c1, c2)
    # d2 = diff_images(c3, c2)

    # print(d1)
    # print(d2)
    # return

    return


if __name__ == "__main__":


    main()
