import os

class BoundingBox(object):

    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def clip_2d_array(self, image_yx_array):
        return image_yx_array[self.min_y:self.max_y+1,
               self.min_x:self.max_x+1]

class Config(object):
    EXTRACTED_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'extracted_images')
    SCREENSHOTS_PATH = os.path.join(os.path.dirname(__file__), 'screenshots')
    CARD_DATA_PATH = os.path.join(os.path.dirname(__file__), 'card_data')

    CARD_WIDTH_PIXELS = 35
    CARD_HEIGHT_PIXELS = 50


    KNN_N_NEIGHBORS = 1
    KNN_N_JOBS = -1

    # min_x, max_x
    PLAYER_DIMENSIONS = [
        BoundingBox(min_x=320, max_x=365, min_y=590, max_y=700)

    ]