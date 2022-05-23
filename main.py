import cv2 as cv    # pip install opencv-python
import numpy as np  # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
from sklearn.cluster import KMeans
from collections import Counter
import PIL


def show_img_and_comparison(img, img_2):
    """Displays 2 images in comparison (Image x Top Colors). 1 row, 2 columns."""

    # create the layout
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,20))

    # adjust the display properties
    ax[0].imshow(img)
    ax[1].imshow(img_2)
    ax[0].axis('off')   # hide the axis
    ax[1].axis('off')   # hide the axis

    fig.tight_layout()
    plt.show()


def get_average_color(img):
    """Finds the average pixel values. Worthless and wrong."""

    img_temp = img.copy()
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(img, axis=(0,1)) # R, G, B

    show_img_and_comparison(img, img_temp)


def get_highest_pixel(img):
    """Counts the number of occurrences per pixel value."""

    img_temp = img.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]

    show_img_and_comparison(img, img_temp)


def load_image(path):

    # load the images
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # resize the images
    dim = (400, 400)    # dimensions of image
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    return img


