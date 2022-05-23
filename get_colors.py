import cv2 as cv    # pip install opencv-python
import numpy as np  # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
from sklearn.cluster import KMeans  # pip install -U scikit-learn
from collections import Counter
import PIL

class GetColors:

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


    def palette(clusters):
        """Creates an image of 50x300px to display as a pallette assigned for each cluster."""

        # make the palette display
        height = 50
        width = 300
        palette = np.zeros((height, width, 3), dtype=np.uint8)

        steps = width / clusters.cluster_centers_.shape[0]
        for i, centers in enumerate(clusters.cluster_centers_): 
            palette[:, int(i * steps):(int((i + 1) * steps)), :] = centers
        
        return palette


    def palette_by_percent(k_cluster):
        """Creates an image of 50x300px to display as a pallette assigned for each cluster."""
        
        height = 50
        width = 300
        palette = np.zeros((height, width, 3), dtype=np.uint8)
        
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
        hex_percent_sorted = {i: j for i, j in sorted(hex_percent.items(), key=lambda item:float(item[0]))}

        # logging purposes
        print(k_cluster.cluster_centers_)
        # print(percent)
        # print(hex_list)
        # print(hex_percent)
        print(hex_percent_sorted)

        return palette


    def rgb_to_hex(r, g, b):

        return "#" + ('{:X}{:X}{:X}').format(r, g, b)


    img = load_image("img/img_1.jpg")


    # set the clusters
    clstr = KMeans(n_clusters=10) # get top 10 colors
    clstr.fit(img.reshape(-1, 3))

    clstr_1 = clstr.fit(img.reshape(-1, 3))
    show_img_and_comparison(img, palette(clstr_1))

    clstr_1 = clstr.fit(img.reshape(-1, 3))
    show_img_and_comparison(img, palette_by_percent(clstr_1))