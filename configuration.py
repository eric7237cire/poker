import os

class BoundingBox(object):

    def __init__(self, min_x=None, max_x=None, min_y=None, max_y=None):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def clip_2d_array(self, image_yx_array):

        return image_yx_array[
               int(self.min_y):int(self.max_y)+1,
               int(self.min_x):int(self.max_x)+1]

class Config(object):
    EXTRACTED_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'extracted_images')
    SCREENSHOTS_PATH = os.path.join(os.path.dirname(__file__), 'screenshots')
    CARD_DATA_PATH = os.path.join(os.path.dirname(__file__), 'card_data')
    NUMBER_DATA_PATH = os.path.join(os.path.dirname(__file__), 'number_data')

    CARD_WIDTH_PIXELS = 35
    CARD_HEIGHT_PIXELS = 50

    ZYNGA_WINDOW = BoundingBox(min_x=8, min_y=320, max_y=-78)

    # All these locations are after slicing off the ZYNGA_WINDOW
    HERO_PLAYER_HOLE_CARDS_LOC = BoundingBox(min_x=320, max_x=365, min_y=290, max_y=340)

    HERO_BETTING_AREA = BoundingBox(min_y=230, max_y=270, min_x=340, max_x=420)
    BETS_AREA = BoundingBox(min_y=75, max_y=325, min_x=100, max_x=650)