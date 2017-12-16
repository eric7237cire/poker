import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from skimage import measure
import os
import cv2
from configuration import Config as cfg
from PIL import Image
import numpy as np
from card_util import *
from PIL import Image, ImageDraw

from scipy.misc import imresize
from matplotlib.path import Path

file_path = os.path.join(
    cfg.CARD_DATA_PATH,
    '2h.png'
)

#image = cv2.imread(file_path)
image = Image.open(file_path)

grey_scale = image.convert('L')

grey_array = np.array(grey_scale)

grey_array = imresize(arr=grey_array, size=(cfg.CARD_HEIGHT_PIXELS, cfg.CARD_WIDTH_PIXELS))

#grey_array[grey_array < 150] = 0
#grey_array[grey_array >= 150] = 255

#http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html?highlight=find_contours
all_contours = measure.find_contours(grey_array, 200)

contours = []

points = generate_points_list(cfg.CARD_WIDTH_PIXELS, cfg.CARD_HEIGHT_PIXELS)

for contour in all_contours:
    min_x, min_y = np.min(contour, axis=0)
    max_x, max_y = np.max(contour, axis=0)

    width = max_x - min_x
    height = max_y - min_y
    if width < 5 or width > 15:
        continue

#    img = Image.new('L', (cfg.CARD_WIDTH_PIXELS, cfg.CARD_HEIGHT_PIXELS), 0)

 #   contour.flatten()

    boolean_mask = extract_polygon_mask_from_contour(
        contour,
        height=cfg.CARD_HEIGHT_PIXELS, width=cfg.CARD_WIDTH_PIXELS,
        all_grid_points_list=points
    )

    extracted_image = extract_image_with_mask(
        grey_array, boolean_mask, background_color=255)
    #ImageDraw.Draw(img).polygon(contour.flatten(), outline=1, fill=1)
    #mask = np.array(img)

    show_image_and_contour(extracted_image, contour)
    #path = Path(contour)
    #grid = path.contains_points(grey_array)

    print(f"Found contour {min_x} {max_x} {min_y} {max_y}\n{contour}")

    contours.append(contour)


# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(grey_array, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

if False:
    image = cv2.imread(file_path)

    bw = get_black_and_white_image(image)

    cnts = find_contours(bw)

    for idx, contour in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        #if cv2.contourArea(contour) < 10:
         #   continue

        x, y, w, h = cv2.boundingRect(contour)

        print(f"Found contour {x} {y} {w} {h}")

        clip_and_save(
            p_orig_image=image,
            x=x,
            y=y,
            w=cfg.CARD_WIDTH_PIXELS,
            h=cfg.CARD_HEIGHT_PIXELS,
            file_name=f"sub_image_{idx:04}.png"
        )

    cv2.imshow("Image", bw)
    #cv2.imshow("Image", image)
    cv2.waitKey(0)