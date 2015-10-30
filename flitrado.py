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
import math

def caja(n, m):
    img = mpimg.imread('imagenes/escgaus.bmp', True)
    result = np.ones((n,m))
    result =  result /(n*m)
    result = ndimage.convolve(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/t.bmp')
def ex(x, n, sigma):
    return math.exp( (-1/2) * ( (x-((n-1)/2))/ sigma)**2 )

def gaus(sigma_1, sigma_2, n, m):
    img = mpimg.imread('imagenes/escgaus.bmp', True)
    ac = 0
    rx = np.ones((1,n))
    ry = np.ones((m,1))

    for i in range(0, n/2):
        ac = ac + ex(i, n, sigma_1)
        rx[0][i] = ex(i, n, sigma_1) * ac**(-1)
        rx[0][n-i-1] = ex(i, n, sigma_1) * ac**(-1)


    ac = 0
    for i in range(0, m/2):
        ac = ac + ex(i, m, sigma_2)
        ry[i][0] = ex(i, m, sigma_2) * ac**(-1)
        ry[m-i-1][0] = ex(i, m, sigma_2) * ac**(-1)


    t = rx*ry
    print t
    print np.sum(t)
    t = t/ np.sum(t)

    result = ndimage.convolve(img, t, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/ver.bmp')



    p = ndimage.gaussian_filter(img, (4,2))
    resultado = Image.fromarray(p.astype(np.uint8))
    resultado.save('resultados/42.bmp')



gaus(4,2,20,10)


# caja(5,5)
