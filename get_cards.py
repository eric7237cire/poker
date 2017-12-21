# import the necessary packages
import logging
import os
import shutil
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import time

from card_classifier import CardClassifier
from card_util import get_game_area_as_2d_array, find_contours_with_cv, \
    get_black_and_white_image, init_logger, clip_and_save
from configuration import Config as cfg
from get_screenshot import capture_screenshot
from number_reader import NumberReader
from scipy.special import comb
logger = logging.getLogger(__name__)
trace_logger = logging.getLogger(__name__ + "_trace")



def get_out_odds(len_common_cards, n_outs, n_chances):
    cards_left = 52.0 - len_common_cards - 2

    denom = comb(N=cards_left, k=n_chances, exact=False, repetition=False)
    num = comb(N=n_outs, k=n_chances, exact=False, repetition=False)

    if n_chances == 2:
        num += n_outs * (cards_left-n_outs)
    return 100.0 * (num / denom)


class GameInfo(object):

    def __init__(self):
        self.common_cards = []
        self.hole_cards = []

        self.pot_starting = None
        self.to_call = None
        self.pot = None






def get_hole_cards(game_area_image_array, card_classifier, game_info):
    image_array = cfg.HERO_PLAYER_HOLE_CARDS_LOC.clip_2d_array(game_area_image_array)

    # show_image(image_array)

    cropped_image = Image.fromarray(image_array)

    game_info.hole_cards = card_classifier.evaluate_hole_card_image(cropped_image)

    logger.info("Found hole card {} and {}".format(
        card_classifier.get_card_string(game_info.hole_cards[0]),
        card_classifier.get_card_string(game_info.hole_cards[1])
    ))


def extract_game_info_from_screenshot(screenshot_file_path, card_classifier, number_reader=None):
    gi = GameInfo()

    game_area_image_array = get_game_area_as_2d_array(screenshot_file_path)

    if number_reader is not None:

        bets = number_reader.get_bets(game_area_image_array.copy())
        gi.to_call = 0

        if len(bets) > 0:
            gi.chips_remaining = number_reader.get_hero_chips_remaining(game_area_image_array.copy())
            gi.to_call = min(np.max(bets[1:]), gi.chips_remaining+bets[0]) - bets[0]

        gi.pot_starting = number_reader.get_starting_pot(game_area_image_array.copy())

        gi.pot = gi.pot_starting + np.sum( [ min(gi.chips_remaining+bets[0], b) for b in bets])

        logger.info(f"Starting Pot: {gi.pot_starting}\nTotal Pot: {gi.pot}\nTo Call: {gi.to_call}")

    get_hole_cards(game_area_image_array=game_area_image_array,
                   card_classifier=card_classifier,
                   game_info=gi)

    image = cv2.imread(screenshot_file_path)

    image = image[400:600, 275:600]  # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # display_cv2_image_with_contours(image, [])

    bw = get_black_and_white_image(image)
    cnts = find_contours_with_cv(bw)

    image_copy = image.copy()

    image_array = np.array(image)

    def get_contour_sort_key(p_contour):
        x, y, w, h = cv2.boundingRect(p_contour)
        return x

    cnts = sorted(cnts, key=get_contour_sort_key)

    # loop over the contours individually
    for idx, contour in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(contour) < 10:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < cfg.CARD_WIDTH_PIXELS - 5 or w > cfg.CARD_WIDTH_PIXELS + 15:
            continue

        if h < cfg.CARD_HEIGHT_PIXELS - 5 or h > cfg.CARD_HEIGHT_PIXELS + 15:
            continue

        clip_and_save(
            p_orig_image=image_copy,
            x=x,
            y=y,
            w=cfg.CARD_WIDTH_PIXELS,
            h=cfg.CARD_HEIGHT_PIXELS,
            file_name=f"sub_image_{idx:04}.png"
        )

        crop_img = image_array[y:y + cfg.CARD_HEIGHT_PIXELS + 1,
                   x:x + cfg.CARD_WIDTH_PIXELS + 1]
        card_image = Image.fromarray(crop_img)

        c = card_classifier.evaluate_card(card_image)

        logger.info(f"Classified extracted image #{idx} as {card_classifier.get_card_string(c)}")

        if c is not None:
            gi.common_cards.append(c)

    # display_cv2_image_with_contours(bw, cnts)
    return gi


def main():
    init_logger()

    card_classifier = CardClassifier()
    number_reader = NumberReader()

    try:
        os.makedirs(cfg.SCREENSHOTS_PATH, exist_ok=True)
        shutil.rmtree(cfg.EXTRACTED_IMAGES_PATH, ignore_errors=True)
        os.makedirs(cfg.EXTRACTED_IMAGES_PATH, exist_ok=True)
    except Exception as ex:
        print(ex)

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d__%H_%M_%S_%f")

    #file_path = os.path.join(cfg.UNIT_TEST_DATA_DIR, 'bet7.png')
    file_path = None

    #if file_path is None:
    iterations = 60 * 60

    for i in range(0, iterations):

        file_path = os.path.join(cfg.SCREENSHOTS_PATH, 'screenshot_{}.png'.format(formatted_time))
        capture_screenshot("chrome", output_file_path=file_path)

        extract_game_info_from_screenshot(file_path, card_classifier, number_reader)


if __name__ == '__main__':
    main()
