# import the necessary packages

from card_classifier import *


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

        hero_bet_grey_array[hero_bet_grey_array >= 121]  = 255

        contour_list = find_contours(grey_array=hero_bet_grey_array,
                                     **cfg.BET_CONTOUR_CONFIG
                                     )
        sorted_contours = sorted(contour_list, key=lambda x: x.bounding_box.min_x)

        #display_image_with_contours(hero_bet_grey_array, [c.points_array for c in sorted_contours])

        self.training_data.extend(zip(hero_numbers, sorted_contours))

    def _digit_contours_to_integer(self, digit_contours):

        digit_contours = sorted(digit_contours, key=lambda x: x.bounding_box.min_x)

        numbers_found = []

        for digit_index, digit_contour in enumerate(digit_contours):

            if digit_index > 0 and digit_contours[digit_index-1].polygon.contains(digit_contour.polygon):
                # Catch inner circle of 0s
                continue

            card_diffs = [diff_polygons(digit_contour, t[1]) for t in self.training_data]
            idx = np.argmin(card_diffs, axis=0)

            numbers_found.append(self.training_data[idx][0])

        print(f"Numbers found: {numbers_found}")

        this_bet_value = None

        if numbers_found:
            this_bet_value = int("".join([str(n) for n in numbers_found if n >= 0]))

        return this_bet_value

    def get_starting_pot(self, game_area_image_array):

        pot_image_array = cfg.STARTING_POT_AREA.clip_2d_array(game_area_image_array)
        # get just green component

        pot_image_grey_array = rgb_yx_array_to_grayscale(pot_image_array)

        digit_contours = find_contours(grey_array=pot_image_grey_array,
                                       **cfg.POT_CONTOUR_CONFIG,
                                       display=False

                                       )

        digit_contours = list(digit_contours)
        starting_pot_value = self._digit_contours_to_integer(digit_contours)

        # display_image_with_contours(pot_image_grey_array, [c.points_array for c in digit_contours])

        #display_image_with_contours(pot_image_grey_array, [digit_contours[2].points_array] +
        #
        #                                                    [x[1].points_array for x in self.training_data if x[0] in [9,0]])

        return starting_pot_value

    def get_bets(self, game_area_image_array):
        image_array = game_area_image_array

        bet_image_array = cfg.BETS_AREA.clip_2d_array(image_array)
        # get just green component

        # display_image_with_contours(bet_image_array, [])

        image_array = bet_image_array[:, :, 1].copy()

        image_array[image_array < 200] = 0

        bet_bubbles = find_contours(grey_array=image_array,
                                    min_width=30,
                                    max_width=100,
                                    min_height=9,
                                    max_height=15
                                    )

        bet_bubbles = sorted(bet_bubbles, key=lambda x: x.bounding_box.min_x)
        # display_image_with_contours(bet_image_array, [c[0] for c in contours])

        all_bets = []

        for contour in bet_bubbles:
            just_text = contour.bounding_box.clip_2d_array(bet_image_array)

            just_text_grey_array = rgb_yx_array_to_grayscale(just_text)

            digit_contours = find_contours(grey_array=just_text_grey_array,
                                           **cfg.BET_CONTOUR_CONFIG
                                           )

            this_bet_value = self._digit_contours_to_integer(digit_contours)

            if this_bet_value is not None:
                all_bets.append(this_bet_value)
            # display_image_with_contours(just_text_grey_array, [c.points_array for c in digit_contours])

        return all_bets
