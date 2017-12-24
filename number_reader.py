# import the necessary packages

import os
import logging
from configuration import Config as cfg
from card_util import get_game_area_as_2d_array, rgb_yx_array_to_grayscale, \
    find_contours, diff_polygons, display_image_with_contours, timeit
import numpy as np

logger = logging.getLogger(__name__)
trace_logger = logging.getLogger(__name__ + "_trace")

class NumberReader(object):

    def __init__(self):
        self.training_data = []

        self.train_numbers(file_path=os.path.join(cfg.NUMBER_DATA_PATH, 'numbers_1_to_6.png'),
                           hero_numbers=[1, 2, 3, 4, 5, 6, -1])

        self.train_numbers(file_path=os.path.join(cfg.NUMBER_DATA_PATH, 'numbers_6_to_0.png'),
                           hero_numbers=[6, 7, 8, 9, 0, -1])

    def train_numbers(self, file_path, hero_numbers):
        image_array = get_game_area_as_2d_array(file_path)

        # display_image_with_contours(image_array, [])

        hero_bet_array = cfg.HERO_BETTING_AREA.clip_2d_array(image_array)

        hero_bet_grey_array = rgb_yx_array_to_grayscale(hero_bet_array)

        hero_bet_grey_array[hero_bet_grey_array >= 121] = 255

        contour_list = find_contours(grey_array=hero_bet_grey_array,
                                     **cfg.OTHER_PLAYER_BET_CONTOUR_CONFIG
                                     )
        sorted_contours = sorted(contour_list, key=lambda x: x.bounding_box.min_x)

        # display_image_with_contours(hero_bet_grey_array, [c.points_array for c in sorted_contours])

        self.training_data.extend(zip(hero_numbers, sorted_contours))

    def _digit_contours_to_integer(self, digit_contours):

        digit_contours = sorted(digit_contours, key=lambda x: x.bounding_box.min_x)

        numbers_found = []

        for digit_index, digit_contour in enumerate(digit_contours):

            if digit_index > 0 and digit_contours[digit_index - 1].polygon.contains(digit_contour.polygon):
                # Catch inner circle of 0s
                continue

            card_diffs = [diff_polygons(digit_contour, t[1]) for t in self.training_data]
            idx = np.argmin(card_diffs, axis=0)

            numbers_found.append(self.training_data[idx][0])

        logger.debug(f"Numbers found: {numbers_found}")

        # Last number will be the $
        numbers_found = numbers_found[0:-1]

        this_bet_value = None

        if numbers_found:
            this_bet_value = int("".join([str(n) for n in numbers_found if n >= 0]))

        return this_bet_value

    def get_starting_pot(self, game_area_image_array):

        pot_image_array = cfg.STARTING_POT_AREA.clip_2d_array(game_area_image_array)

        pot_image_grey_array = rgb_yx_array_to_grayscale(pot_image_array)

        digit_contours = find_contours(grey_array=pot_image_grey_array,
                                       **cfg.POT_CONTOUR_CONFIG,
                                       display=False

                                       )

        digit_contours = list(digit_contours)

        # display_image_with_contours(pot_image_grey_array, [c.points_array for c in digit_contours])

        starting_pot_value = self._digit_contours_to_integer(digit_contours)

        # display_image_with_contours(pot_image_grey_array, [digit_contours[2].points_array] +
        #
        #                                                    [x[1].points_array for x in self.training_data if x[0] in [9,0]])

        if starting_pot_value is None:
            starting_pot_value = 0

        return starting_pot_value

    @timeit
    def get_hero_chips_remaining(self, game_area_image_array):
        chips_image_array = cfg.HERO_REMAINING_CHIPS_AREA.clip_2d_array(game_area_image_array)
        #display_image_with_contours(chips_image_array, contours=[])

        chips_image_grey_array = rgb_yx_array_to_grayscale(chips_image_array)

        digit_group_contours = find_contours(grey_array=chips_image_grey_array,
                                       **cfg.CHIPS_REMAINING_DIGIT_GROUPS_CONTOUR_CONFIG,
                                       display=False

                                       )

        digit_group_contours = list(digit_group_contours)

        chips_image_grey_array = self.add_spaces_to_digits(
            image_grey_array=chips_image_grey_array,
            digit_group_contours=digit_group_contours
        )

        digit_contours = find_contours(grey_array=chips_image_grey_array,
                                             **cfg.CHIPS_REMAINING_DIGIT_CONTOUR_CONFIG,
                                             display=False

                                             )
        digit_contours = list(digit_contours)
        chips_remaining = self._digit_contours_to_integer(digit_contours)

        logger.info(f"Chips remaining: {chips_remaining}")

        if False:
            display_image_with_contours(chips_image_grey_array,
                                    [c.points_array for c in digit_contours])

        return chips_remaining

    def add_spaces_to_digits(self, image_grey_array, digit_group_contours, digit_width=5, fill_color=0):

        if digit_group_contours is None or len(digit_group_contours) <= 1:
            logger.warning("Not enough digit groups")
            return

        right_x = np.max(digit_group_contours[-1].points_array[:, 1])
        left_x = np.min(digit_group_contours[0].points_array[:, 1])

        if digit_width == 6:
            right_x = int(round(right_x+0.4))
        else:
            right_x = int(round(right_x))
        left_x = int(round(left_x))

        # account for the $ sign and an extra space
        x_to_insert_blank_line = right_x - digit_width - 1

        seps_added = 0
        while x_to_insert_blank_line > left_x:
            image_grey_array = np.insert(image_grey_array, obj=x_to_insert_blank_line,
                                               values=fill_color, axis=1)

            x_to_insert_blank_line -= digit_width
            seps_added += 1

            # each group of 3 gets an extra space
            if seps_added % 3 == 0:
                if digit_width == 6:
                    x_to_insert_blank_line -= 3
                else:
                    x_to_insert_blank_line -= 1

        return image_grey_array

    @timeit
    def get_bets(self, game_area_image_array):

        bet_image_array = cfg.BETS_AREA.clip_2d_array(game_area_image_array)
        #display_image_with_contours(bet_image_array, [])
        # get just green component

        # display_image_with_contours(bet_image_array, [])

        # Basically we just want green things, so...

        # take max of red and blue columns (indexs 0 and 2)
        max_red_blue_value = np.max(bet_image_array[:, :, [0, 2]], axis=2)

        # build a boolean array of green values that are less than the max of red or blue
        cond = bet_image_array[:, :, 1] < max_red_blue_value

        # for those pixels where green not the max, set it to 0, otherwise subtract the max
        bet_image_array[:, :, 1] = np.where(cond, 0, bet_image_array[:, :, 1] - max_red_blue_value)

        # now we just have a picture of green things
        image_array = bet_image_array[:, :, 1].copy()

        bet_bubbles = find_contours(grey_array=image_array,
                                    min_width=30,
                                    max_width=100,
                                    min_height=9,
                                    # Sometimes green chips can make height larger
                                    max_height=35,
                                    #display=True
                                    )

        bet_bubbles = sorted(bet_bubbles, key=lambda x: x.bounding_box.min_x)

        #display_image_with_contours(image_array, [b.points_array for b in bet_bubbles])

        all_bets = [0] * 5

        center_bet_area_yx = bet_image_array.shape[0] / 2, bet_image_array.shape[1] / 2

        for contour in bet_bubbles:
            # logger.info(contour.bounding_box)
            just_text = contour.bounding_box.clip_2d_array(image_array)

            center_bet_yx = list(contour.bounding_box.center_yx())
            center_bet_yx[0] -= center_bet_area_yx[0]
            center_bet_yx[1] -= center_bet_area_yx[1]

            player_position = None
            if abs(center_bet_yx[0]) < 75 and abs(center_bet_yx[1]) < 15:
                player_position = 0
            elif center_bet_yx[0] > 0 and center_bet_yx[1] < 0:
                player_position = 1
            elif center_bet_yx[0] < 0 and center_bet_yx[1] < 0:
                player_position = 2
            elif center_bet_yx[0] < 0 and center_bet_yx[1] > 0:
                player_position = 3
            elif center_bet_yx[0] > 0 and center_bet_yx[1] > 0:
                player_position = 4
            else:
                raise Exception("cain")

            # clip off 4 leftmost pixels which are giving false contours
            just_text_grey_array = just_text[:, 4:]

            digit_group_contours = find_contours(grey_array=just_text_grey_array,
                                                 **cfg.OTHER_PLAYER_BET_DIGIT_GROUP_CONFIG,
                                                 display=False
                                                 )

            digit_group_contours = list(digit_group_contours)

            #display_image_with_contours(just_text_grey_array, [c.points_array for c in digit_group_contours])

            bet_image_grey_array = self.add_spaces_to_digits(
                image_grey_array=just_text_grey_array,
                digit_group_contours=digit_group_contours,
                digit_width=6,
                fill_color=255
            )

            digit_contours = find_contours(grey_array=bet_image_grey_array,
                                           **cfg.OTHER_PLAYER_BET_CONTOUR_CONFIG,
                                           display=False
                                           )

            digit_contours = list(digit_contours)

            # display_image_with_contours(bet_image_grey_array, [c.points_array for c in digit_contours])

            this_bet_value = self._digit_contours_to_integer(digit_contours)

            if this_bet_value is not None:
                logger.info(f"Found bet {this_bet_value}.  Players position: {player_position}")
                logger.debug(
                     "Bet area center: {}.  Bet center: {} ".format(
                        center_bet_area_yx,
                         center_bet_yx))


                #display_image_with_contours(bet_image_grey_array, [c.points_array for c in digit_contours])
                all_bets[player_position] = this_bet_value

        return all_bets
