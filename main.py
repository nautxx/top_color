import argparse
import cv2 as cv    # pip install opencv-python
import numpy as np  # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
from sklearn.cluster import KMeans  # pip install -U scikit-learn
from collections import Counter
import PIL


def load_image(path):

    # load the image
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # resize the image
    dim = (500, 500)    # set dimensions of the image
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    return img


def rgb_to_hex(r, g, b):

    return "#" + ('{:X}{:X}{:X}').format(r, g, b)


def show_img_and_comparison(img, img_2):
    """Displays 2 images in comparison (Image x Top Colors). 1 row, 2 columns."""

    # create the layout
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 12.5))

    # adjust the plot display properties
    ax[0].imshow(img)
    ax[1].imshow(img_2)
    ax[0].axis('off')   # hide the axis
    ax[1].axis('off')   # hide the axis

    fig.canvas.manager.set_window_title('Top Color') 
    fig.tight_layout()
    plt.show()


def get_average_color(img):
    """Finds the average pixel values. Worthless and wrong."""

    average_color = img.copy()
    average_color[:, :, 0], average_color[:, :, 1], average_color[:, :, 2] = np.average(img, axis=(0, 1)) # R, G, B

    average_color_pixel = average_color[0][0]
    average_color_pixel_hex = rgb_to_hex(average_color_pixel[0], average_color_pixel[1], average_color_pixel[2])
    
    print(f"Average color: {average_color_pixel_hex}")

    return average_color


def get_top_color(img):
    """Counts the number of occurrences per pixel value."""

    top_color = img.copy()
    unique, counts = np.unique(top_color.reshape(-1, 3), axis=0, return_counts=True)
    top_color[:, :, 0], top_color[:, :, 1], top_color[:, :, 2] = unique[np.argmax(counts)]

    top_color_pixel = top_color[0][0]
    top_color_pixel_hex = rgb_to_hex(top_color_pixel[0], top_color_pixel[1], top_color_pixel[2])
    
    print(f"Top color: {top_color_pixel_hex}")
    
    return top_color


def palette(clusters):
    """Creates an image of 80x250px to display as a pallette assigned for each cluster."""

    # make the palette display
    height = 50
    width = 500
    palette = np.zeros((height, width, 3), dtype=np.uint8)

    steps = width / clusters.cluster_centers_.shape[0]
    for i, centers in enumerate(clusters.cluster_centers_): 
        palette[:, int(i * steps):(int((i + 1) * steps)), :] = centers
    
    return palette


def palette_by_percent(k_cluster):
    """Creates an image of 80x250px to display as a pallette assigned for each cluster."""
    
    height = 80
    width = 250
    palette = np.zeros((height, width, 3), dtype=np.uint8)  # create the palette bar
    
    pixel_count = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)    # count how many pixels per cluster
    
    percent = {}
    for i in counter:
        percent[i] = np.round(counter[i] / pixel_count, 2)

    percent = dict(sorted(percent.items()))

    hex_list = []
    step = 0
    for i, centers in enumerate(k_cluster.cluster_centers_):
        r = int(k_cluster.cluster_centers_[i][0])
        g = int(k_cluster.cluster_centers_[i][1])
        b = int(k_cluster.cluster_centers_[i][2])

        palette[:, step:int(step + percent[i] * width + 1), :] = centers
        step += int(percent[i] * width + 1) # adjusts width of pallette by percentage

        hex_list.append(rgb_to_hex(r, g, b))

    hex_percent = dict(zip(percent.values(), hex_list))
    hex_percent_sorted = {
        i: j for i, j in sorted(hex_percent.items(), key=lambda item:float(item[0]))
    }

    print(f"Top colors: {hex_percent_sorted}")

    return palette


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A simple python script to get the most common colors of an image file.'
    )
    parser.add_argument("--file_path", "-fp", help="The name of the file to load.", type=str, default="img/img_1.jpg")
    parser.add_argument("--colors", "-c", help="Number of top colors.", type=int, default=5)
    args = parser.parse_args()

    img = load_image(args.file_path)

    get_average_color(img)
    get_top_color(img)

    # set the clusters
    clstr = KMeans(n_clusters=args.colors)
    clstr_ = clstr.fit(img.reshape(-1, 3))

    show_img_and_comparison(img, palette_by_percent(clstr_))