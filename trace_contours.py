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

IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'images')

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#cv2.fillPoly( np.zeros(p_orig_image.shape, dtype=np.uint8), np.array([[[0,353], [0,382], [6, 382], [6, 353]]]), color = [255,255,255])[0:6, 353:382]

#cv2.fillPoly( np.zeros( (7, 7, 3), dtype=np.uint8), np.array([[[0,3], [0,5], [6, 5], [6, 3]]]), color = [255,255,255])[6]

def clip_and_save(p_orig_image, x,y,w,h, file_name):
    """

    :param p_orig_image:  cv2 image with dimensions [y][x][RGB] = 0-255
    :param contour_to_crop:
    :param file_name:
    :return:
    """

    os.makedirs(IMAGES_PATH, exist_ok=True)


    crop_img = p_orig_image[ y:y+h+1, x:x+w+1]

    # save the result
    cv2.imwrite(os.path.join(IMAGES_PATH, file_name), crop_img)



def main():

    try:
        shutil.rmtree(IMAGES_PATH, ignore_errors=True)
        os.makedirs(IMAGES_PATH)
    except Exception as ex:
        print(ex)

    sys.argv.extend(['-i', 'screenshot.png', ])

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")

    args = vars(ap.parse_args())

    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # blur = cv2.GaussianBlur(gray,(1,1),1000)
    blur = gray
    flag, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    # edged = cv2.Canny(thresh, 50, 100)
    # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)
    edged = thresh

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)


    orig = image.copy()

    # loop over the contours individually
    for idx, contour in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(contour) < 10:
            continue

        x, y, w, h = cv2.boundingRect(contour)


        if w < 25 or w > 50:
            continue

        if h < 30 or h > 70:
            continue

        clip_and_save(
            p_orig_image=orig,
            x=x,
            y=y,
            w=w,
            h=h,
            file_name=f"sub_image_{idx:04}.png"
        )
        # compute the rotated bounding box of the contour

        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))





    # show the output image
    #cv2.imshow("Image", thresh)
    cv2.imshow("Image", orig)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
