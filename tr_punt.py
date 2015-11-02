import os,sys
import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cv2
from scipy.misc import imread


###Primera parte


def histogram(img_file):
    img = mpimg.imread(img_file, True);
    counts, bins, bars = plt.hist(img.flatten(), bins=255, color='gray', range=(0, 255));
    plt.show();


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

    img = mpimg.imread('imagenes/escilum.tif')
    hist, binds, c = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))
    result = tr_punt(binds, [51, 150], [0,255])
    im2 = np.interp(img.flatten(), binds ,result)
    res_img = im2.reshape(img.shape)
    img = Image.fromarray(res_img.astype(np.uint8))
    img.save('resultados/second.png')

    plt.clf()
    plt.hist(result, bins=255, color='yellow', range=(0,255))
    plt.show()


def third():
    img = mpimg.imread('imagenes/escilum.tif')

    hist, bins, b = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))
    cdf = hist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    im2 = np.interp(img.flatten(), bins[:-1], cdf)

    res_img = im2.reshape(img.shape)

    img = Image.fromarray(res_img.astype(np.uint8))
    img.save('resultados/third.png')
    plt.clf()

    plt.hist(cdf, bins=255, color='yellow', range=(0,255))
    plt.show()

def eq_hist_quad(N, M):
    img = imread('imagenes/escilum.tif')
    im = Image.open('imagenes/car.jpg').convert('L')
    size = im.size
    row = size[1]
    col = size[0]
    hist, bins, b = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))


histogram('imagenes/escilum.tif')
