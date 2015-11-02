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


##
# @param{hist}  an image's histogram
# @param{input_range}
# @param{output_range}
# This methods change and histogram range for other giveb
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


def problem_2(imgName, new_img):
    img = mpimg.imread(imgName);
    hist, binds, c = plt.hist(img.flatten(), bins=255, range=(0, 255));

    #change histogram range
    result = tr_punt(binds, [51, 150], [0,255]);

    #create a new image after changed its histogram values.
    im2 = np.interp(img.flatten(), binds ,result);
    res_img = im2.reshape(img.shape);
    img = Image.fromarray(res_img.astype(np.uint8));
    img.save(new_img);

    plt.clf()
    #show new histogram
    plt.hist(result, bins=255, color='yellow', range=(0,255))
    plt.show()



def eq_hist(hist):
    cdf = hist.cumsum(); # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]; # normalize the value
    return cdf;

def third(imgName, newImgName):
    img = mpimg.imread(imgName, True);
    hist, bins, b = plt.hist(img.flatten(), bins=255, range=(0, 255));
    cdf = eq_hist(hist);

    # create the new image
    im2 = np.interp(img.flatten(), bins[:-1], cdf)
    res_img = im2.reshape(img.shape)
    img = Image.fromarray(res_img.astype(np.uint8))
    img.save(newImgName)
    plt.clf()

    # show the new histogram
    plt.hist(cdf, bins=255, color='green', normed= True, range=(0,255))
    plt.show()

# def eq_hist_quad(N, M):
#     img = imread('imagenes/escilum.tif')
#     im = Image.open('imagenes/car.jpg').convert('L')
#     size = im.size
#     row = size[1]
#     col = size[0]
#     hist, bins, b = plt.hist(img.flatten(), bins=255, color='yellow', range=(0, 255))


#histogram('imagenes/escilum.tif')
# problem_2('imagenes/escilum.tif', 'resultados/problem_2.png');
# third('imagenes/escilum.tif', 'resultados/problem_3.png');
# third('resultados/problem_3_bis.png', 'resultados/problem_3_bis2.png');
