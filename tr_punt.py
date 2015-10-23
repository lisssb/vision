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
    def tr_punt(hist, input_range=[], output_range=[]):
        if len(input_range) != 2 or len(output_range) != 2:
            print "The input_range and the output_range must be specified and its length must be 2"
            return

        result = [];
        for i in hist:
            if i < input_range[0]:
                value = 0
            elif i > input_range[1]:
                value = 255
            else:
                value = (i-input_range[0]) * output_range[1]/(input_range[1] - input_range[0])
            result.append(value)
        return result



    img = mpimg.imread('escilum.tif')
    hist, binds, c = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))
    result = tr_punt(binds, [51, 150], [0,255])
    im2 = np.interp(img.flatten(), binds ,result)
    res_img = im2.reshape(img.shape)
    img = Image.fromarray(res_img.astype(np.uint8))
    img.save('second.png')


def third():
    img = mpimg.imread('escilum.tif')

    hist, bins, b = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))
    cdf = hist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    im2 = np.interp(img.flatten(), bins[:-1], cdf)

    res_img = im2.reshape(img.shape)

    img = Image.fromarray(res_img.astype(np.uint8))
    img.save('third.png')

second()
third()
