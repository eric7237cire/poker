import logging
import os
import sys

import cv2
import imutils
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.path import Path
from shapely.affinity import translate, scale
from skimage import measure
import time
from configuration import Config as cfg
from dto import BoundingBox, Contour

logger = logging.getLogger(__name__)


def init_logger():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.ERROR)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

    if False:
        logging.getLogger("card_classifier_trace").setLevel(logging.INFO)
        logging.getLogger("number_reader").setLevel(logging.INFO)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logger.debug('%r  %2.2f ms' % \
                     (method.__name__, (te - ts) * 1000))
        return result

    return timed


def diff_polygons(contour_1, contour_2, scale_polygons=True):
    """

    :return: Total of non intersecting area
    """

    if contour_1 is None or contour_2 is None:
        return 10000000000000

    poly1 = contour_1.polygon
    poly2 = contour_2.polygon

    if not poly1.is_valid or not poly2.is_valid:
        logger.warning("Polygons not valid")
        return 10000000000000

    minx1, miny1, maxx1, maxy1 = poly1.bounds
    minx2, miny2, maxx2, maxy2 = poly2.bounds

    width1 = maxx1 - minx1
    width2 = maxx2 - minx2
    height1 = maxy1 - miny1
    height2 = maxy2 - miny2

    if scale_polygons:
        poly2 = scale(geom=poly2, xfact=width1 / width2, yfact=height1 / height2, origin='centroid')

    poly1 = translate(poly1, xoff=-poly1.bounds[0], yoff=-poly1.bounds[1])
    poly2 = translate(poly2, xoff=-poly2.bounds[0], yoff=-poly2.bounds[1])

    intersecting_area = poly1.intersection(poly2).area

    return poly1.area + poly2.area - 2 * intersecting_area


def display_image_with_contours(grey_array, contours):
    # Display the image and plot all contours found
    fig, ax = plt.subplots()

    if grey_array is not None:
        ax.imshow(grey_array, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def rgb_yx_array_to_grayscale(array):
    image = Image.fromarray(array)
    grey_image = image.convert('L')
    return np.array(grey_image)


def card_to_grayscale_2d_array(image):
    grey_scale = image.convert('L')

    grey_array = np.array(grey_scale)

    # grey_array = imresize(arr=grey_array, size=(cfg.CARD_HEIGHT_PIXELS, cfg.CARD_WIDTH_PIXELS))

    return grey_array


def trim_main_window_image_array(image_array):
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


def get_game_area_as_2d_array(screenshot_file_path):
    image = Image.open(screenshot_file_path)

    image_array = np.array(image)

    return trim_main_window_image_array(image_array)


def find_contours(
        grey_array, min_width=5, max_width=15,
        min_height=5, max_height=100,
        value_threshold=150,
        fully_connected="low",
        display=False
):
    """

    :return:  iterable of contours in card
    """

    # grey_array[grey_array < 150] = 0
    # grey_array[grey_array >= 150] = 255

    if grey_array is None or grey_array.ndim != 2:
        logger.warning("Not a valid array passed to find_contours")
        return

    # http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html?highlight=find_contours
    all_contours = measure.find_contours(grey_array, level=value_threshold, fully_connected=fully_connected)

    # Todo find inner shapes and subtract from polygon
    contour_list = []

    for points_array in all_contours:

        b = BoundingBox()
        b.min_y, b.min_x = np.min(points_array, axis=0)
        b.max_y, b.max_x = np.max(points_array, axis=0)

        c = Contour()
        c.bounding_box = b

        if not np.array_equal(points_array[0], points_array[-1]):
            points_array = np.append(points_array, np.expand_dims(points_array[0], axis=0), axis=0)

        c.set_points_array(points_array)

        contour_list.append(c)

    contour_list = sorted(contour_list, key=lambda x: x.bounding_box.min_x)

    if display:
        display_image_with_contours(grey_array, [c.points_array for c in contour_list])

    for idx, c in enumerate(contour_list):

        if c is None:
            continue

        width = c.bounding_box.max_x - c.bounding_box.min_x
        height = c.bounding_box.max_y - c.bounding_box.min_y

        if width < min_width or width > max_width:
            # logger.debug(f"Skipping contour #{idx}: {c} due to width")
            continue

        if height < min_height or height > max_height:
            # logger.debug(f"Skipping contour #{idx}: {c} due to height")
            continue

        # print(f"Found contour @ {min_x},{min_y} Width={width} Height={height} Numpoints={len(contour)}")

        if display:
            # display_image_with_contours(grey_array, [c.points_array ])
            pass

        if not c.polygon.is_valid:
            logger.warning("Polygon is not valid")
            continue

        # See if any additional contours fit 100% inside
        for idx2 in range(idx + 1, len(contour_list)):
            c2 = contour_list[idx2]

            if c2 is None:
                continue

            if c2.polygon is not None and c.polygon.contains(c2.polygon) and c2.polygon.is_valid:
                c.polygon = c.polygon.difference(c2.polygon)
                # don't return it in future runs
                contour_list[idx2] = None
            elif c2.bounding_box.min_x > c.bounding_box.max_x:
                break

        yield c


def generate_points_list(width, height):
    """
    returns a 2d array, x,y of all points in an
    integer grid of width/height

    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    return points


def extract_polygon_mask_from_contour(contour, width, height, all_grid_points_list):
    """

    :param contour: a y/x list of points
    :return: An image of height x width with True where the pixel is in the
    polygon defined by the contour
    """

    # https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    contour_xy = contour[:, [1, 0]]

    # https://matplotlib.org/api/path_api.html#matplotlib.path.Path
    path = Path(contour_xy)
    grid = path.contains_points(all_grid_points_list, radius=-1)
    grid = grid.reshape((height, width))

    return grid


def extract_image_with_mask(image, boolean_mask, background_color):
    """

    :param image: 2d numpy array, y x
    :param boolean_mask: same size as image, True to extract
    :param background_color what color to set where boolean_mask is False
    :return: copy of image
    """
    r_image = image.copy()

    r_image[np.logical_not(boolean_mask)] = background_color

    return r_image


def clip_and_save(p_orig_image, x, y, w, h, file_name):
    """

    :param p_orig_image:  cv2 image with dimensions [y][x][RGB] = 0-255
    :param contour_to_crop:
    :param file_name:
    :return:
    """

    os.makedirs(cfg.EXTRACTED_IMAGES_PATH, exist_ok=True)

    crop_img = p_orig_image[y:y + h + 1, x:x + w + 1]

    # save the result
    cv2.imwrite(os.path.join(cfg.EXTRACTED_IMAGES_PATH, file_name), crop_img)
