import cv2
import imutils
import os
import matplotlib.pyplot as plt
from datetime import datetime
from configuration import Config as cfg
import numpy as np
from matplotlib.path import Path
from scipy.misc import imresize
from skimage import measure
import matplotlib.patches as patches
from dto import BoundingBox, Contour

def display_cv2_image_with_contours(grey_array, contours):
    # Display the image and plot all contours found
    fig, ax = plt.subplots(1)

    if grey_array is not None:
        ax.imshow(grey_array, interpolation='bicubic', cmap=plt.cm.gray)

    #ax2 = fig.add_subplot(111, aspect='equal')
    for n, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        ax.add_patch(patches.Rectangle(
            xy=(x,y),
            width=w,
            height=h,
            fill=False,
            linewidth=1,
            edgecolor='g'
        ))


    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

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


def card_to_grayscale_2d_array(image):
    grey_scale = image.convert('L')

    grey_array = np.array(grey_scale)

    # grey_array = imresize(arr=grey_array, size=(cfg.CARD_HEIGHT_PIXELS, cfg.CARD_WIDTH_PIXELS))

    return grey_array


def find_contours_in_card(
        grey_array, min_width=5, max_width=15,
        min_height=5, max_height = 100,
        value_threshold=150
):
    """

    :param image: PIL image
    :return:  iterable of contours in card, matching the
    number and symbol
    """

    # grey_array[grey_array < 150] = 0
    # grey_array[grey_array >= 150] = 255

    # http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html?highlight=find_contours
    all_contours = measure.find_contours(grey_array, value_threshold)

    for contour in all_contours:

        b = BoundingBox()
        b.min_y, b.min_x = np.min(contour, axis=0)
        b.max_y, b.max_x = np.max(contour, axis=0)

        width = b.max_x - b.min_x
        height = b.max_y - b.min_y

        if width < min_width or width > max_width:
            continue

        if height < min_height or height > max_height:
            continue

        #print(f"Found contour @ {min_x},{min_y} Width={width} Height={height} Numpoints={len(contour)}")
        if not np.array_equal(contour[0],contour[-1]):
            contour = np.append(contour, np.expand_dims(contour[0], axis=0), axis=0)

        c  = Contour()
        c.points_array, c.bounding_box = contour, b
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


def get_contour_xy(contour):
    """
    Contours are in y,x
    :param contour:
    :return: same points x,y
    """
    # https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    contour_xy = contour[:, [1, 0]]
    return contour_xy


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


def show_image(image):
    """

    :param image: 2d array, first dimension y
    :return:
    """
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def show_image_and_contour(image, contour):
    """

    :param image: 2d array, first dimension y
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.imshow(image, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def find_contours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    return cnts


def get_black_and_white_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Will set everything below 200 to black, above to 255 (maxval/white)
    flag, thresh = cv2.threshold(gray, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

    return thresh


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
