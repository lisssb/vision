import os,sys
import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import cv2
from scipy.misc import imread
from scipy import ndimage
from array import array

def caja(n, m):
    img = mpimg.imread('imagenes/escgaus.bmp', True)
    im = Image.open("imagenes/escgaus.bmp")
    w = im.size[0];
    h = im.size[1];

    med = (n - 1) / 2;
    current = 0;
    result = np.zeros((h,w))

    for i in range(med, w-med):
        for j in range (med, h-med):
            for t in range (1, med+1):
                for k in range (1, med+1):
                    current += float(img[i-t][j-k]) + float(img[i-t][j+k])
                    current += float(img[i][j-k]) + float(img[i][j+k])
                    current += float(img[i+t][j]) + float(img[i-t][j])
                    current += float(img[i+t][j-k]) + float(img[i+t][j+k])
            current += float(img[i][j])
            result[i][j] = current/(n*n)
            current = 0

    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/t.bmp')


caja(3,3)
