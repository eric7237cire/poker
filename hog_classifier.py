import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

import os
import cv2
from configuration import Config as cfg
from PIL import Image
import numpy as np

from scipy.misc import imresize

file_path = os.path.join(
    cfg.CARD_DATA_PATH,
    '2h.png'
)

#image = cv2.imread(file_path)
image = Image.open(file_path)

#It's possible to use LA to also have an alpha channel
#http://pillow.readthedocs.io/en/4.3.x/handbook/concepts.html#concept-modes
grey_scale = image.convert('L')

grey_array = np.array(grey_scale)

grey_array = imresize(arr=grey_array, size=(cfg.CARD_HEIGHT_PIXELS, cfg.CARD_WIDTH_PIXELS))

#http://scikit-image.org/docs/0.13.x/api/skimage.feature.html#skimage.feature.hog
fd, hog_image = hog(
    image=grey_array, orientations=8, pixels_per_cell=(2, 20),

    cells_per_block=(1, 1),
    block_norm = "L1",
    visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
