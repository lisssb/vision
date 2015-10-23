import os,sys
import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cv2
from scipy.misc import imread


###Primera parte


def first():
    img = mpimg.imread('escilum.tif')
    counts, bins, bars = plt.hist(img.flatten(), 255, normed=True, color='red')
    plt.show()


def second():
    def tr_punt(im, input_range=[], output_range=[]):
        img = Image.open(im).convert("L")
        if len(input_range) != 2 or len(output_range) != 2:
            print "The input_range and the output_range must be specified and its length must be 2"
            return

        def change_pixel (i):
            if i < input_range[0]:
                value = 0
            elif i > input_range[1]:
                value = 255
            else:
                value = (i-input_range[0]) * output_range[1]/(input_range[1] - input_range[0])
            return value

        result = img.point(change_pixel)
        result.save('second.png')
        plt.hist(imread('second.png').flatten(), bins=255, color='yellow', range=(0, 255))
        plt.show()

    result = tr_punt('escilum.tif', [51, 150], [0,255])


def third():
    img = mpimg.imread('escilum.tif')

    hist, bins, b = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))
    cdf = hist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    im2 = np.interp(img.flatten(), bins[:-1], cdf)

    res_img = im2.reshape(img.shape)

    img = Image.fromarray(res_img.astype(np.uint8))
    img.save('third.png')

third()
