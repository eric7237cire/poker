import os

class Config(object):
    EXTRACTED_IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'extracted_images')
    SCREENSHOTS_PATH = os.path.join(os.path.dirname(__file__), 'screenshots')
    CARD_DATA_PATH = os.path.join(os.path.dirname(__file__), 'card_data')

    CARD_WIDTH_PIXELS = 35
    CARD_HEIGHT_PIXELS = 50