import unittest

import os

from card_classifier import CardClassifier
from number_reader import NumberReader
from card_util import init_logger
from get_cards import extract_game_info_from_screenshot, get_out_odds, perc_to_odds_to_1, GameInfo



class TestOdds(unittest.TestCase):

    def test_out_perc(self):

        self.assertAlmostEqual(2.13, get_out_odds(3, 1, 1), places=2)
        self.assertAlmostEqual(31.45, get_out_odds(3, 8, 2), places=2)
        self.assertAlmostEqual(28.26, get_out_odds(4, 13, 1), places=2)

        self.assertAlmostEqual(36.17, get_out_odds(3, 17, 1), places=3)
        self.assertAlmostEqual(36.9565, get_out_odds(4, 17, 1), places=3)
        self.assertAlmostEqual(59.7594, get_out_odds(3, 17, 2), places=3)

        self.assertAlmostEqual(19.15, get_out_odds(3, 9, 1), places=2)
        self.assertAlmostEqual(19.57, get_out_odds(4, 9, 1), places=2)
        self.assertAlmostEqual(34.97, get_out_odds(3, 9, 2), places=2)


    def test_perc_to_odds(self):

        odds = perc_to_odds_to_1(36.96)
        self.assertAlmostEqual(1.7, odds, places=1)


        odds = perc_to_odds_to_1(48.1)
        self.assertAlmostEqual(1.1, odds, places=1)

    def test_pot_odds(self):

        gi  = GameInfo()
        gi.pot = 100
        gi.to_call = 50

        self.assertEqual(2, gi.pot_odds())