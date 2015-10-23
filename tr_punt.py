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





#b = img[...,0]  # blue channel
#g = img[...,1]  # green channel
#r = img[...,2]  # red channel
#plt.hist(b, bins=100, normed=True, histtype='step')
#plt.hist(g, bins=100, normed=True, histtype='step')
#plt.hist(r, bins=100, normed=True, histtype='step')



#print bins
#plt.show()
