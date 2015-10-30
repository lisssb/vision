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
import timeit

def caja_convolve1(n, m):
    img = imread('imagenes/escgaus.bmp', True)
    result = np.ones(n)
    result = result/n
    result = ndimage.convolve1d(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/caja_convolve1.bmp')
    

def caja(n, m):
    img = imread('imagenes/escgaus.bmp', True)
    result = np.ones((n,m))
    result =  result /(n*m)
    result = ndimage.convolve(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/caja.bmp')


def ex(x, n, sigma):
    return math.exp( (-1/2) * ( (x-((n-1)/2))/ sigma)**2 )

def gaus(sigma_1, sigma_2, n, m):
    img = imread('imagenes/escimp5.bmp', True)
    ac = 0
    rx = np.ones((1,n))
    ry = np.ones((m,1))

    for i in range(0, n):
        ac += ex(i, n, sigma_1)
        rx[0][i] = ex(i, n, sigma_1) * ac**(-1)

    ac = 0
    for i in range(0, m):
        ac += ex(i, m, sigma_2)
        ry[i][0] = ex(i, m, sigma_2) * ac**(-1)

    t = rx*ry
    t = t/ np.sum(t)

    result = ndimage.convolve(img, t, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/ver.bmp')

    #
    #
    # p = ndimage.gaussian_filter(img, (1,2))
    # resultado = Image.fromarray(p.astype(np.uint8))
    # resultado.save('resultados/4444444444444.bmp')

def seven():
    n = 7
    m = 7

    img = imread('imagenes/checker.bmp', True)
    result = np.ones((n,m))
    result =  result /(n*m)
    result = ndimage.convolve(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/franja_caja.bmp')

    n=10
    m=10
    sigma_1 = 2
    sigma_2 = 2

    ac = 0
    rx = np.ones((1,n))
    ry = np.ones((m,1))

    for i in range(0, n):
        ac += ex(i, n, sigma_1)
        rx[0][i] = ex(i, n, sigma_1) * ac**(-1)

    ac = 0
    for i in range(0, m):
        ac += ex(i, m, sigma_2)
        ry[i][0] = ex(i, m, sigma_2) * ac**(-1)

    t = rx*ry
    t = t/ np.sum(t)

    result = ndimage.convolve(img, t, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/franja_gausiano.bmp')


# caja(5,5)
# caja_convolve1(5,5)

if __name__=='__main__':
    from timeit import Timer
    t1 = Timer("caja(3,3)", "from __main__ import caja")
    t2 = Timer("caja_convolve1(3,3)", "from __main__ import caja_convolve1")
    print(t1.timeit(1), "caja")
    print(t2.timeit(1), "1d")
