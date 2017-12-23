# import the necessary packages
import logging
import os
import shutil
from datetime import datetime
import poker
import cv2
import numpy as np
from PIL import Image
import time
from threading import Event
from card_classifier import CardClassifier
from card_util import get_game_area_as_2d_array, find_contours_with_cv, \
    get_black_and_white_image, init_logger, clip_and_save
from configuration import Config as cfg
from get_screenshot import capture_screenshot
from number_reader import NumberReader
import sys
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


def perc_to_odds_to_1(perc):
    left = 100 - perc
    right = left / perc

    return right

class GameInfo(object):

    def __init__(self):
        self.common_cards = []
        self.hole_cards = []

        self.pot_starting = None
        self.to_call = None
        self.pot = None
        self.chips_remaining = None

    def pot_odds(self):
        if self.pot is None or self.to_call is None or self.to_call <= 0:
            return -2
        return self.pot / self.to_call

    def is_equal(self, other_gi):

        if other_gi is None:
            return False

        for atts in ['pot_starting', 'to_call', 'pot']:
            if getattr(self, atts) != getattr(other_gi, atts):
                return False

        if not np.array_equal(self.common_cards, other_gi.common_cards):
            return False

        if not np.array_equal(self.hole_cards, other_gi.hole_cards):
            return False

        return True


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
    logger.info(f"Starting catpure of {screenshot_file_path}")
    gi = GameInfo()

    game_area_image_array = get_game_area_as_2d_array(screenshot_file_path)

    if number_reader is not None:

        bets = number_reader.get_bets(game_area_image_array.copy())
        gi.to_call = 0
        gi.chips_remaining = number_reader.get_hero_chips_remaining(game_area_image_array.copy())

        if len(bets) > 0 and gi.chips_remaining is not None:
            gi.to_call = min(np.max(bets[1:]), gi.chips_remaining+bets[0]) - bets[0]

        gi.pot_starting = number_reader.get_starting_pot(game_area_image_array.copy())

        if gi.chips_remaining is not None:
            gi.pot = gi.pot_starting + np.sum(  [ min(gi.chips_remaining+bets[0], b) for b in bets])

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

event_exit = Event()

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


    last_gi = None
    for i in range(0, iterations):


        file_path = os.path.join(cfg.SCREENSHOTS_PATH, 'screenshot_{}.png'.format(formatted_time))
        capture_screenshot("chrome", output_file_path=file_path)

        gi = extract_game_info_from_screenshot(file_path, card_classifier, number_reader)

        if gi.is_equal(last_gi):
            #event_exit.wait(.5)
            for i in range(0, 30):
                time.sleep(0.1)
            continue

        print("*" * 80)
        print(" " * 80)
        for c in gi.common_cards:
            print(f"Common card: {card_classifier.get_card_string(c)}")
        for h in gi.hole_cards:
            print(f"Hole card: {card_classifier.get_card_string(h)}")

        if gi.chips_remaining is not None:
            print("Chips remaining: {:,}".format(gi.chips_remaining))
        print("Starting Pot: {:,}".format(gi.pot_starting))
        if gi.pot is not None:
            print("Pot: {:,}".format(gi.pot))
        print("To Call: {:,}".format(gi.to_call))

        for outs in [9, 8, 4, 2]:
            perc = get_out_odds(len_common_cards=len(gi.common_cards), n_outs=outs, n_chances=1)
            ratio = perc_to_odds_to_1(perc)
            print(f"{outs} outs.  {perc:.2f}% = {ratio:.2f}:1")

            if len(gi.common_cards) == 3:
                perc = get_out_odds(len_common_cards=len(gi.common_cards), n_outs=outs, n_chances=2)
                ratio = perc_to_odds_to_1(perc)

                print(f"{outs} outs.  2 cards to go {perc:.2f}% = {ratio:.2f}:1")

        print(f"\nPot odds: { 100.0/(1+gi.pot_odds()):.2f}%  = {gi.pot_odds():.2f}:1 ")

        if len(gi.hole_cards) == 2 and gi.hole_cards[0] is not None:
            hole_card_string = "".join([card_classifier.get_card_short_string(hc) for hc in gi.hole_cards])
            common_cards_string = "".join([card_classifier.get_card_short_string(cc) for cc in gi.common_cards])
            equity3 = poker.run_simulation(3, hole_card_string, common_cards_string, 500000, False)
            equity4 = poker.run_simulation(4, hole_card_string, common_cards_string, 500000, False)
            equity5 = poker.run_simulation(5, hole_card_string, common_cards_string, 500000, False)

            print(f"Equity:\n3 players: {equity3:.2f}%\n4 players: {equity4:.2f}%\n5 players: {equity5:.2f}% ")

        last_gi = gi

        for i in range(0,30):
            time.sleep(0.1)


def quit(signo, _frame):
    print("Interrupted by %d, shutting down" % signo)
    event_exit.set()

if __name__ == '__main__':

    import signal

    #signal.signal(signal.CTRL_C_EVENT, quit)
    #signal.signal(signal.CTRL_BREAK_EVENT, quit)
    signal.signal(signal.SIGINT,quit)
    signal.signal(signal.SIGBREAK, quit)

    try:
        main()
    except KeyboardInterrupt:
        pass
