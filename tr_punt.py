import os,sys
import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cv2


img = mpimg.imread('escilum.tif')
im = Image.open('escilum.tif')
counts, bins, bars = plt.hist(img.flatten(), bins=300, normed=True)


#plt.imshow(img, cmap=cm.Greys_r)



#b = img[...,0]  # blue channel
#g = img[...,1]  # green channel
#r = img[...,2]  # red channel
#plt.hist(b, bins=100, normed=True, histtype='step')
#plt.hist(g, bins=100, normed=True, histtype='step')
#plt.hist(r, bins=100, normed=True, histtype='step')



#print bins
#plt.show()

def tr_punt(hist, input_range=[], output_range=[]):
    if len(input_range) != 2 or len(output_range) != 2:
        print "The input_range and the output_range must be specified and its length must be 2"
        return

    result = []
    for i in hist:
        if i < input_range[0]:
            result.append(0)
        elif i > input_range[1]:
            result.append(255)
        else:
            result.append( (i-input_range[0]) * output_range[1]/(input_range[1] - input_range[0]) )
    return result


r = tr_punt(bins, [51, 150], [0, 255])
plt.clf()
plt.title('Histogram of escilum')
plt.grid(True)
plt.hist(r,  bins=150, color='blue', normed=True)
plt.clf()
#plt.show()

def ver (i):
    if i < 51:
        p =  0
    elif i > 150:
        p =  255
    else:
        p =  ((i-51) * 255)/(150 - 51)
    return p

tt = im.point(ver)
tt.save("contraste-10.tif")
plt.hist(np.asarray(tt).flatten(), bins=150, color='green', normed=True)


#plt.imshow(img, cmap=cm.Greys_r)
plt.show()
