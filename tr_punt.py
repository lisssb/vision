import os,sys
import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cv2


img = mpimg.imread('escilum.tif')
p = plt.hist(img.flatten(), bins=300, normed=True)


#b = img[...,0]  # blue channel
#g = img[...,1]  # green channel
#r = img[...,2]  # red channel
#plt.hist(b, bins=100, normed=True, histtype='step')
#plt.hist(g, bins=100, normed=True, histtype='step')
#plt.hist(r, bins=100, normed=True, histtype='step')

plt.title('Histogram of escilum')
plt.grid(True)
plt.show()



#def tr_punt(hist, input_range, output_range):
