import unittest
import os
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
from get_cards import *

class TestGetCards(unittest.TestCase):

    def setUp(self):

        self.longMessage = True
        self.UNIT_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'unit_test_data')


    def test_get_cards(self):

        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_5cards_1.png')
        card_classifier = CardClassifier()

        gi = extract_cards_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('2','h'), msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('3', 's'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('t', 'c'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('k', 'h'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('7', 'c'))

    def test_get_cards_blocked(self):

        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_blocked_cards.png')
        card_classifier = CardClassifier()

        gi = extract_cards_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('5','h'), msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('j', 'c'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('7', 's'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('8', 'h'))

    def test_get_hole_cards(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards.png')
        card_classifier = CardClassifier()

        gi = extract_cards_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('q', 's'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('3', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('3', 'd'))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('8', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('t', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

    def test_get_hole_cards_2(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_2.png')
        card_classifier = CardClassifier()

        gi = extract_cards_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('3', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('6', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('9', 'h'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('4', 'h'))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('5', 'c'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('5', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))
