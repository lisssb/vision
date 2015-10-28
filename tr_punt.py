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

    plt.clf()
    plt.hist(result, bins=255, color='yellow', range=(0,255))
    plt.show()


def third():
    img = mpimg.imread('escilum.tif')

    hist, bins, b = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))
    cdf = hist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    im2 = np.interp(img.flatten(), bins[:-1], cdf)

    res_img = im2.reshape(img.shape)

    img = Image.fromarray(res_img.astype(np.uint8))
    img.save('third.png')
    plt.clf()

    plt.hist(cdf, bins=255, color='yellow', range=(0,255))
    plt.show()

def eq_hist_quad(N, M):
    img = imread('escilum.tif')
    mid_value = round(M*N/2)
    in_ = 0
    for i in range(1, M):
        for j in range(1, N):
            in_ += 1
            if(in_ == mid_val):
                padM = i -1
                padN = j -1
                break
    print mid_val
    #padarray
    B = np.pad(img, [padM, padN])
    for i in range(1, size(B, 1) - (padM*2)+1):
        cdf = [0] * 256
        inc = 1
        for x in range(1, M):
            for y in range(1,N):
                #finde the middle element in the WINDOW
                if(inc == mid_val):
                    ele=B(i+x-1,j+y-1)+1

                pos=B(i+x-1,j+y-1)+1
                cdf[pos]=cdf[pos]+1
                inc=inc+1
        for l in range (2, 256):
            cdf[l] = cdf[l] + cdf[l-1]

        Img(i, j) = round(cdf[ele])/(M*N)*255

# def eq_hist_quad():
#     img = cv2.imread('escilum.tif',0)
#
#     # create a CLAHE object (Arguments are optional).
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     cl1 = clahe.apply(img)
#
#     cv2.imwrite('clahe_2.jpg',cl1)

eq_hist_quad(3,3)
# first()
# second()
#third()
