# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from configuration import Config as cfg

class Card(object):

    def __init__(self, card_index, card_file_name, card_image):

        self.card_index = card_index
        self.card_file_name = card_file_name
        self.card_image = card_image

        self.pixels = None
        self.hist = None

def image_to_feature_vector(image):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, (cfg.CARD_WIDTH_PIXELS, cfg.CARD_HEIGHT_PIXELS)).flatten()


def preprocess(image):
    img = cv2.resize(image, (cfg.CARD_WIDTH_PIXELS, cfg.CARD_HEIGHT_PIXELS))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2 )
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
    blur_thresh = cv2.GaussianBlur(thresh,(5,5),5)
    return blur_thresh

def diff_images(img1, img2):
    #cv2.imshow("Image1", img1)
    #cv2.imshow("Image2", img2)
    p_img1 = preprocess(img1)
    p_img2 = preprocess(img2)
    # cv2.imshow("Image", image_copy)

    diff = cv2.absdiff(p_img1, p_img2)


    #cv2.imshow("P Image1", p_img1)
    #cv2.imshow("P Image2", p_img2)
    #cv2.imshow("diff", diff)

    #diff = cv2.GaussianBlur(diff,(5,5),5)
    #flag, diff = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
    diff_sum = np.sum(diff)


    #cv2.imshow("diff2", diff)
   # print (diff_sum)
    #cv2.waitKey(0)

    return diff_sum



def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def main():
    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imagePaths = list(paths.list_images(cfg.CARD_DATA_PATH))

    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    rawImages = []
    features = []
    labels = []
    cards = []

    test_cards = []
    cards_to_eval = []

    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']
    suits = ['d', 'c', 's', 'h']

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)

        file_name = os.path.basename(imagePath)

        # File name should be like ac for Ace of clubs or 2d for 2 of diamonds
        label = ranks.index(file_name[0]) + 13 * suits.index(file_name[1])

        print("Label is {} for file name {}".format(label, file_name))

        card = Card(card_index=label, card_file_name=file_name, card_image=image)

        if file_name[2] == '_':
            cards_to_eval.append(card)
        else:
            test_cards.append(card)

        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)

        card.pixels = pixels
        card.hist = hist

        cards.append(card)

        # update the raw images, features, and labels matricies,
        # respectively
        rawImages.append(pixels)
        features.append(hist)
        labels.append(label)

        # show an update every 1,000 images
        #if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

    for card in cards_to_eval:
        print(f"Evaluating {card.card_file_name} / {card.card_index} ")

        card_diffs = [diff_images(card.card_image, t.card_image) for t in test_cards]

        index = np.argmin(card_diffs, axis=0)

        print(f"Closest to {test_cards[index].card_file_name}")

    return

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    rawImages = np.array(rawImages)
    features = np.array(features)
    labels = np.array(labels)
    print("[INFO] pixels matrix: {:.2f}MB".format(
        rawImages.nbytes / (1024 * 1000.0)))
    print("[INFO] features matrix: {:.2f}MB".format(
        features.nbytes / (1024 * 1000.0)))

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.25, random_state=42)
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
        features, labels, test_size=0.25, random_state=42)

    # train and evaluate a k-NN classifer on the raw pixel intensities
    print("[INFO] evaluating raw pixel accuracy...")
    model = KNeighborsClassifier(n_neighbors=cfg.KNN_N_NEIGHBORS,
                                 n_jobs=cfg.KNN_N_JOBS)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)

    for card in cards:
        prediction = model.predict([card.pixels])
        print(f"Prediction for {card.card_file_name}/{card.card_index} is {prediction}")

    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

    # train and evaluate a k-NN classifer on the histogram
    # representations
    print("[INFO] evaluating histogram accuracy...")
    model = KNeighborsClassifier(n_neighbors=cfg.KNN_N_NEIGHBORS,
                                 n_jobs=cfg.KNN_N_JOBS)
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)

    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

    for card in cards:
        prediction = model.predict([card.hist])
        print(f"Prediction for {card.card_file_name}/{card.card_index} is {prediction}")




if __name__ == "__main__":
    main()