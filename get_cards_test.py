import unittest

import os

from card_classifier import CardClassifier
from number_reader import NumberReader
from card_util import init_logger
from get_cards import extract_game_info_from_screenshot
from PIL import Image
import numpy as np

card_classifier = CardClassifier()

number_reader = NumberReader()


class TestGetCards(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_logger()

    def setUp(self):
        self.longMessage = True
        self.UNIT_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'unit_test_data')

    def _get_gi(self,file_path):
        image = Image.open(file_path)

        image_array = np.array(image)

        return extract_game_info_from_screenshot(screenshot_image_rgb_yx_array=image_array,
                                                 card_classifier=card_classifier,
                                                 number_reader=number_reader)

    def test_get_cards(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_5cards_1.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('2', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('3', 's'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('t', 'c'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('k', 'h'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('7', 'c'))

    def test_get_cards_2(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_5cards_2.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('7', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('6', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('6', 's'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('5', 'h'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('4', 'c'))

    def test_get_3cards(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_3cards.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('j', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('2', 'd'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('7', 'c'))

    def test_get_cards_blocked(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_blocked_cards.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('5', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('j', 'c'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('7', 's'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('8', 'h'))

    def test_get_hole_cards(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('q', 's'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('3', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('3', 'd'))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('8', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('t', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

    def test_get_hole_cards_2(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_2.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('3', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('6', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('9', 'h'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('4', 'h'))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('5', 'c'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('5', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

    def test_get_hole_cards_3(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_3.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('2', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('9', 'd'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('8', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[2]))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('9', 'c'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('7', 'h'))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('t', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('j', 'c'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

    def test_get_hole_cards_4(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_4.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('6', 'c'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('9', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('2', 'c'),
                         msg=card_classifier.get_card_string(gi.common_cards[2]))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('a', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('j', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

    def test_get_hole_cards_5(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_5.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('t', 's'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('k', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('j', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[2]))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('2', 'c'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('3', 'd'))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('6', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('9', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

    def test_bet(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'bet.png'))


        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('3', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('4', 's'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('8', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[2]))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('7', 'c'))

        self.assertEqual(4, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('a', 'c'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('q', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(12000, gi.to_call)
        self.assertEqual(90000, gi.pot_starting)
        self.assertEqual(90000 + 12000, gi.pot)

    def test_bet2(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'bet2.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('6', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('q', 'd'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('4', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[2]))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('a', 'h'))

        self.assertEqual(4, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('9', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('a', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(1000, gi.to_call)
        self.assertEqual(2000, gi.pot_starting)
        self.assertEqual(2000 + 1000 + 2000, gi.pot)

    def test_bet3(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'bet3.png'))

        self.assertEqual(0, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('k', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('q', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(200246 - 4123, gi.to_call)
        self.assertEqual(0, gi.pot_starting)
        self.assertEqual(200246 + 2000 + 4123, gi.pot)

    def test_bet4(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'bet4.png'))

        self.assertEqual(0, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('a', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('k', 'c'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(-4579 + 2000, gi.to_call)
        self.assertEqual(0, gi.pot_starting)
        self.assertEqual(4579 + 4000, gi.pot)

    def test_bet5(self):
        """
        Tests pot amounts with side pots
        """
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'bet5.png'))

        self.assertEqual(0, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('5', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('6', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        # self.assertEqual(492000 - 2000, gi.to_call)
        self.assertEqual(398000, gi.to_call)
        self.assertEqual(0, gi.pot_starting)
        # self.assertEqual(394000 + 492000 + 2000 + 1000, gi.pot)
        self.assertEqual(394000 + 400000 + 2000 + 1000, gi.pot)

    def test_bet6(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'bet6.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('9', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('7', 's'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('4', 's'),
                         msg=card_classifier.get_card_string(gi.common_cards[2]))

        self.assertEqual(3, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('j', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('j', 'c'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(64000, gi.to_call)
        self.assertEqual(1202000, gi.pot_starting)
        self.assertEqual(64000, gi.chips_remaining)
        self.assertEqual(1202000 + 64000, gi.pot)

    def test_bet7(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'bet7.png'))

        self.assertEqual(0, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('7', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('4', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(37044 - 2000, gi.to_call)
        self.assertEqual(0, gi.pot_starting)
        self.assertEqual(1117776, gi.chips_remaining)
        self.assertEqual(2000 + 37044 + 37044, gi.pot)

    def test_cards_1(self):
        gi = self._get_gi(os.path.join(self.UNIT_TEST_DATA_DIR, 'cards1.png'))

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('q', 'c'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('3', 'c'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('a', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[2]))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('4', 's'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('3', 's'))

        self.assertEqual(5, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('3', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('8', 'c'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(-6000, gi.to_call)
        self.assertEqual(14000, gi.pot_starting)
        self.assertEqual(911776, gi.chips_remaining)
        self.assertEqual(14000 + 6000, gi.pot)
