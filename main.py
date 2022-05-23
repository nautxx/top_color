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


