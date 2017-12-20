import unittest

import os

from card_classifier import CardClassifier
from number_reader import NumberReader
from card_util import init_logger
from get_cards import extract_game_info_from_screenshot

card_classifier = CardClassifier()

number_reader = NumberReader()

class TestGetCards(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_logger()



    def setUp(self):
        self.longMessage = True
        self.UNIT_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'unit_test_data')



    def test_get_cards(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_5cards_1.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('2', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('3', 's'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('t', 'c'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('k', 'h'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('7', 'c'))

    def test_get_cards_2(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_5cards_2.png')
        card_classifier = CardClassifier()

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('7', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('6', 'h'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('6', 's'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('5', 'h'))
        self.assertEqual(gi.common_cards[4], card_classifier.get_card_id('4', 'c'))

    def test_get_3cards(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_3cards.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('j', 'd'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('2', 'd'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('7', 'c'))

    def test_get_cards_blocked(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_blocked_cards.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

        self.assertEqual(gi.common_cards[0], card_classifier.get_card_id('5', 'h'),
                         msg=card_classifier.get_card_string(gi.common_cards[0]))
        self.assertEqual(gi.common_cards[1], card_classifier.get_card_id('j', 'c'))
        self.assertEqual(gi.common_cards[2], card_classifier.get_card_id('7', 's'))
        self.assertEqual(gi.common_cards[3], card_classifier.get_card_id('8', 'h'))

    def test_get_hole_cards(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards.png')
        card_classifier = CardClassifier()

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

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

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

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
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_3.png')


        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

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
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_4.png')


        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier)

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
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'screenshot_with_hole_cards_5.png')


        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier
                                               )

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
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'bet.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier,
                                               number_reader=number_reader)

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
        self.assertEqual(90000+12000, gi.pot)

    def test_bet2(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'bet2.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier,
                                               number_reader=number_reader)

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
        self.assertEqual(2000+1000+2000, gi.pot)

    def test_bet3(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'bet3.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier,
                                               number_reader=number_reader)

        self.assertEqual(0, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('k', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('q', 'h'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        self.assertEqual(200246 - 4123, gi.to_call)
        self.assertEqual(0, gi.pot_starting)
        self.assertEqual(200246 + 2000 + 4123, gi.pot)

    def test_bet4(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'bet4.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier,
                                               number_reader=number_reader)

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
        Tests pot amounts are correct if there is a side pot (basically someone
        bet more than we have)
        """
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'bet5.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier,
                                               number_reader=number_reader)

        self.assertEqual(0, len(gi.common_cards))

        self.assertEqual(gi.hole_cards[0], card_classifier.get_card_id('5', 's'),
                         msg=card_classifier.get_card_string(gi.hole_cards[0]))
        self.assertEqual(gi.hole_cards[1], card_classifier.get_card_id('6', 'd'),
                         msg=card_classifier.get_card_string(gi.hole_cards[1]))

        #self.assertEqual(492000 - 2000, gi.to_call)
        self.assertEqual(398000, gi.to_call)
        self.assertEqual(0, gi.pot_starting)
        #self.assertEqual(394000 + 492000 + 2000 + 1000, gi.pot)
        self.assertEqual(394000 + 400000 + 2000 + 1000, gi.pot)

    def test_bet6(self):
        file_path = os.path.join(self.UNIT_TEST_DATA_DIR, 'bet6.png')

        gi = extract_game_info_from_screenshot(screenshot_file_path=file_path, card_classifier=card_classifier,
                                               number_reader=number_reader)

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
