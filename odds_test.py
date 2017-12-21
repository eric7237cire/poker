import unittest

import os

from card_classifier import CardClassifier
from number_reader import NumberReader
from card_util import init_logger
from get_cards import extract_game_info_from_screenshot, get_out_odds



class TestOdds(unittest.TestCase):

    def test_out_perc(self):

        self.assertAlmostEqual(2.13, get_out_odds(3, 1, 1), places=2)
        self.assertAlmostEqual(31.45, get_out_odds(3, 8, 2), places=2)
        self.assertAlmostEqual(28.26, get_out_odds(4, 13, 1), places=2)

        self.assertAlmostEqual(36.17, get_out_odds(3, 17, 1), places=3)
        self.assertAlmostEqual(36.9565, get_out_odds(4, 17, 1), places=3)
        self.assertAlmostEqual(59.7594, get_out_odds(3, 17, 2), places=3)