# import the necessary packages
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


class GameInfo(object):

    def __init__(self):
        self.common_cards = []
        self.hole_cards = []


def get_game_area_as_2d_array(screenshot_file_path):
    image = Image.open(screenshot_file_path)

    image_array = np.array(image)

    image_array = image_array[cfg.ZYNGA_WINDOW.min_y:cfg.ZYNGA_WINDOW.max_y,
                  cfg.ZYNGA_WINDOW.min_x:]

    left_index = 0

    for i in range(0, 300):
        column_sum = np.sum(image_array[:, i])

        if column_sum > 0:
            break

    left_index = i

    image_array = image_array[:, left_index:]

    return image_array


class NumberReader(object):

    def __init__(self):
        self.train_numbers(file_path=os.path.join(cfg.NUMBER_DATA_PATH, 'numbers_1_to_6.png'),
        hero_numbers=[1, 2, 3, 4, 5, 6])

    def train_numbers(self, file_path, hero_numbers):
        image_array = get_game_area_as_2d_array(file_path)

        #display_image_with_contours(image_array, [])

        hero_bet_array = cfg.HERO_BETTING_AREA.clip_2d_array(image_array)

        #display_image_with_contours(hero_bet_array, [])

        hero_bet_image = Image.fromarray(hero_bet_array)
        hero_bet_grey_image = hero_bet_image.convert('L')
        hero_bet_grey_array = np.array(hero_bet_grey_image)

        contours = find_contours_in_card(grey_array=hero_bet_grey_array,
                                         min_width=2,
                                         max_width=14,
                                         min_height=5,
                                         max_height=11,
                                         value_threshold=70
                                         )

        display_image_with_contours(hero_bet_grey_array, [c.points_array for c in contours])


def get_bets(screenshot_file_path):
    image_array = get_game_area_as_2d_array(screenshot_file_path)

    bet_image_array = cfg.BETS_AREA.clip_2d_array(image_array)
    # get just green component

    #display_image_with_contours(bet_image_array, [])

    image_array = bet_image_array[:,:,1].copy()

    image_array[image_array < 200] = 0

    contours = find_contours_in_card(grey_array=image_array,
                                     min_width=30,
                                     max_width=100,
                                     min_height=9,
                                     max_height=15
                                     )

    contours = list(contours)
    #display_image_with_contours(bet_image_array, [c[0] for c in contours])

    for contour, bounding_box in contours:

        just_text = bounding_box.clip_2d_array(bet_image_array)

        just_text_image = Image.fromarray(just_text)
        grey_scale_text = just_text_image.convert('L')
        just_text_grey_array = np.array(grey_scale_text)

        contours = find_contours_in_card(grey_array=just_text_grey_array,
                                         min_width=2,
                                         max_width=12,
                                         min_height=5,
                                         max_height=11,
                                         value_threshold=80
                                         )

        display_image_with_contours(just_text_grey_array, [c[0] for c in contours])

    pass

def get_hole_cards(screenshot_file_path, card_classifier, game_info):

    image_array = get_game_area_as_2d_array(screenshot_file_path)

    image_array = cfg.HERO_PLAYER_HOLE_CARDS_LOC.clip_2d_array(image_array)

    # show_image(image_array)

    cropped_image = Image.fromarray(image_array)

    game_info.hole_cards = card_classifier.evaluate_hole_card_image(cropped_image)

    print("Found hole card {} and {}".format(
        card_classifier.get_card_string(game_info.hole_cards[0]),
        card_classifier.get_card_string(game_info.hole_cards[1])
    ))


def extract_game_info_from_screenshot(screenshot_file_path, card_classifier):
    gi = GameInfo()

    number_read = NumberReader()

    #get_bets(screenshot_file_path)

    #return

    get_hole_cards(screenshot_file_path=screenshot_file_path,
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
    cnts = find_contours(bw)

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

        print(f"Classified extracted image #{idx} as {card_classifier.get_card_string(c)}")

        if c is not None:
            gi.common_cards.append(c)

    # display_cv2_image_with_contours(bw, cnts)
    return gi


def main():
    card_classifier = CardClassifier()

    try:
        os.makedirs(cfg.SCREENSHOTS_PATH, exist_ok=True)
        shutil.rmtree(cfg.EXTRACTED_IMAGES_PATH, ignore_errors=True)
        os.makedirs(cfg.EXTRACTED_IMAGES_PATH, exist_ok=True)
    except Exception as ex:
        print(ex)

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d__%H_%M_%S_%f")

    file_path = os.path.join(cfg.SCREENSHOTS_PATH, 'screenshot_{}.png'.format(formatted_time))
    capture_screenshot("chrome", output_file_path=file_path)

    # file_path = r"E:\git\poker\screenshots\screenshot_2017_12_17__14_54_05_754106.png"

    extract_game_info_from_screenshot(file_path, card_classifier)


if __name__ == '__main__':
    main()
