import os

from dto import BoundingBox


class Config(object):
    EXTRACTED_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'extracted_images')
    SCREENSHOTS_PATH = os.path.join(os.path.dirname(__file__), 'screenshots')
    CARD_DATA_PATH = os.path.join(os.path.dirname(__file__), 'card_data')
    NUMBER_DATA_PATH = os.path.join(os.path.dirname(__file__), 'number_data')
    UNIT_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'unit_test_data')

    CARD_WIDTH_PIXELS = 35
    CARD_HEIGHT_PIXELS = 50

    ZYNGA_WINDOW = BoundingBox(min_x=8, min_y=320, max_y=-78)

    # All these locations are after slicing off the ZYNGA_WINDOW

    HERO_PLAYER_HOLE_CARDS_LOC = BoundingBox(min_x=320, max_x=360, min_y=290, max_y=340)

    HERO_BETTING_AREA = BoundingBox(min_y=240, max_y=260, min_x=350, max_x=420)
    BETS_AREA = BoundingBox(min_y=75, max_y=325, min_x=100, max_x=650)
    HERO_REMAINING_CHIPS_AREA = BoundingBox(min_y=345, max_y=365, min_x=350, max_x=420)

    STARTING_POT_AREA = BoundingBox(min_x=320, max_x=440, min_y=200, max_y=230)

    OTHER_PLAYER_BET_DIGIT_GROUP_CONFIG = {
        "min_width": 2,
        "max_width": 14,
        "min_height": 5,
        "max_height": 11,
        "value_threshold": 80
    }
    OTHER_PLAYER_BET_CONTOUR_CONFIG = {
        "min_width": 2,
        "max_width": 7,
        "min_height": 5,
        "max_height": 9,
        "value_threshold": 70
    }

    POT_CONTOUR_CONFIG = {
        "min_width": 2,
        "max_width": 14,
        "min_height": 5,
        "max_height": 11,
        "value_threshold": 220,
        # Works better probably because the digits are higher valued
        "fully_connected": "high"
    }

    CHIPS_REMAINING_DIGIT_GROUPS_CONTOUR_CONFIG = {
        "min_width": 2,
        "max_width": 64,
        "min_height": 5,
        "max_height": 13,
        "value_threshold": 20,
        # Works better probably because the digits are higher valued
        "fully_connected": "high"
    }

    CHIPS_REMAINING_DIGIT_CONTOUR_CONFIG = {
        "min_width": 2,
        "max_width": 6,
        "min_height": 5,
        "max_height": 13,
        "value_threshold": 180,
        # Works better probably because the digits are higher valued
        "fully_connected": "high"
    }
