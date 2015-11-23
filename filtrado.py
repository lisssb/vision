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



def caja(imgName, n, m=None):
    if m is None:
        m = n
    img = imread(imgName, True);
    result = np.ones((n,m))
    result =  result /(n*m)
    result = ndimage.convolve(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    return resultado

def mediana(imgName, n):
    med = int(n*n/2)
    img = imread(imgName, True)
    def mediana_aux (a):
        b = np.sort(a)
        return b[med]
    result = ndimage.generic_filter(img, mediana_aux, n)
    resultado = Image.fromarray(result.astype(np.uint8))
    return resultado;

def bilateral(im, n, sigmaColor, sigmaSpace):
    img = imread(im)
    result = cv2.bilateralFilter(img, n, sigmaColor, sigmaSpace)
    resultado = Image.fromarray(result.astype(np.uint8))
    return resultado

# bilateral('imagenes/escgaus.bmp')
# mediana('imagenes/escgaus.bmp', 7)

def caja_convolve1(n, m):
    img = imread('imagenes/escgaus.bmp', True)
    result = np.ones(n)
    result = result/n
    result = ndimage.convolve1d(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/caja_convolve1.bmp')

def gaus2(imgName, sigma_1, n):
    img = imread(imgName, True)
    ac = 0
    med = n/2;
    x, y = np.mgrid[-n/2 + 1:n/2 + 1, -n/2 + 1:n/2 + 1]
    g = (1/(2*math.pi*(sigma_1**2))) *  np.exp(-((x**2 + y**2)/(2.0*sigma_1**2)))
    t= g #/is a multiply by pi sigma I an getting the nromalized matriz

    result = ndimage.convolve(img, t, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    return resultado

def ex(x, n, sigma):
    return math.exp( (-1/2) * ( (x-((n-1)/2))/ sigma)**2 )

def Guassian( x, sigma):
    c = 2.0 * sigma * sigma;
    return math.exp(-x * x / c) / math.sqrt(c * math.pi);


def gaus(imgName,  sigma_1, n, sigma_2,  m):
    img = imread(imgName, True)
    med_n = int(n/2)
    med_m = int(m/2)
    rx = np.ones((n, 1))
    ry = np.ones((1, m))
    ac = 0

    for i in range(0, n):
        rx[i][0] = Guassian(i - n/2, sigma_1)

    for i in range(0, m):
        ry[0][i] = Guassian(i - m/2, sigma_2)


    # for i in range(0, med_n):
    #     rx[med_n + i ][0] = med_n - i + 1
    #     rx[med_n - i][0] = med_n - i + 1
    #
    # for i in range(0, med_m):
    #     ry[0][med_m + i ] = med_m - i + 1
    #     ry[0][med_m - i] = med_m - i + 1

    t = rx*ry
    t = t /t.sum()
    print t



    result = ndimage.convolve(img, t, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    rx = rx/rx.sum()
    ry = ry/ry.sum()

    r_aux = ndimage.convolve1d(img, rx  , mode='constant', cval=1.0)
    r_aux = ndimage.convolve1d(r_aux, ry, mode='constant', cval=1.0)
    print r_aux

    return resultado




# caja(5,5)
#caja_convolve1(5,5)

# if __name__=='__main__':
#     from timeit import Timer
#     t1 = Timer("caja('imagenes/escgaus.bmp', 3)", "from __main__ import caja")
#     t2 = Timer("caja_convolve1(3, 3)", "from __main__ import caja_convolve1")
#     print(t1.timeit(1), "caja")
#     print(t2.timeit(1), "1d")

# print kernel2d
def main():
    # caja('imagenes/escgaus.bmp', 3).save('resultados/escgaus_caja_3.png')
    # caja('imagenes/escgaus.bmp', 5).save('resultados/escgaus_caja_5.png')
    # caja('imagenes/escgaus.bmp', 7).save('resultados/escgaus_caja_7.png')
    # caja('imagenes/escgaus.bmp', 9).save('resultados/escgaus_caja_9.png')
    # caja('imagenes/escgaus.bmp', 11).save('resultados/escgaus_caja_11.png')
    # caja('imagenes/escgaus.bmp', 13).save('resultados/escgaus_caja_13.png')
    #
    # caja('imagenes/escimp5.bmp', 3).save('resultados/escimp5_caja_3.png')
    # caja('imagenes/escimp5.bmp', 5).save('resultados/escimp5_caja_5.png')
    # caja('imagenes/escimp5.bmp', 7).save('resultados/escimp5_caja_7.png')
    # caja('imagenes/escimp5.bmp', 9).save('resultados/escimp5_caja_9.png')
    # caja('imagenes/escimp5.bmp', 11).save('resultados/escimp5_caja_11.png')
    # caja('imagenes/escimp5.bmp', 13).save('resultados/escimp5_caja_13.png')
    img = imread('imagenes/escimp5.bmp');
    gaus('imagenes/escimp5.bmp', 5,25,5,25).save('resultados/mio.bmp')
    resultado = Image.fromarray(ndimage.gaussian_filter(img, 5).astype(np.uint8))
    resultado.save('resultados/gausx5.bmp')
    # gaus('imagenes/escgaus.bmp', 1, 5).save('resultados/escgaus_gaus_5.png')
    # gaus('imagenes/escgaus.bmp', 1.4, 7).save('resultados/escgaus_gaus_7.png')
    # gaus('imagenes/escgaus.bmp', 1.8, 9).save('resultados/escgaus_gaus_9.png')
    # gaus('imagenes/escgaus.bmp', 2.2, 11).save('resultados/escgaus_gaus_11.png')
    # gaus('imagenes/escgaus.bmp', 2.6, 13).save('resultados/escgaus_gaus_13.png')

    # gaus('imagenes/escimp5.bmp', 0.6, 3).save('resultados/escimp5_gaus_3.png')
    # gaus('imagenes/escimp5.bmp', 1, 5).save('resultados/escimp5_gaus_5.png')
    # gaus('imagenes/escimp5.bmp', 1.4, 7).save('resultados/escimp5_gaus_7.png')
    # gaus('imagenes/escimp5.bmp', 1.8, 9).save('resultados/escimp5_gaus_9.png')
    # gaus('imagenes/escimp5.bmp', 2.2, 11).save('resultados/escimp5_gaus_11.png')
    # gaus('imagenes/escimp5.bmp', 2.6, 13).save('resultados/escimp5_gaus_13.png')



    # caja('imagenes/checker.bmp', 3).save('resultados/checker_caja_3.png')
    # caja('imagenes/checker.bmp', 5).save('resultados/checker_caja_5.png')
    # caja('imagenes/checker.bmp', 7).save('resultados/checker_caja_7.png')
    # caja('imagenes/checker.bmp', 9).save('resultados/checker_caja_9.png')
    # caja('imagenes/checker.bmp', 11).save('resultados/checker_caja_13.png')
    # caja('imagenes/checker.bmp', 15).save('resultados/checker_caja_15.png')
    # caja('imagenes/checker.bmp', 17).save('resultados/checker_caja_17.png')
    # caja('imagenes/checker.bmp', 19).save('resultados/checker_caja_19.png')
    # caja('imagenes/checker.bmp', 21).save('resultados/checker_caja_21.png')
    #
    # gaus('imagenes/checker.bmp', 0.6, 3).save('resultados/checker_gaus_3.png')
    # gaus('imagenes/checker.bmp', 1, 5).save('resultados/checker_gaus_5.png')
    # gaus('imagenes/checker.bmp', 1.4, 7).save('resultados/checker_gaus_7.png')
    # gaus('imagenes/checker.bmp', 1.8, 9).save('resultados/checker_gaus_9.png')
    # gaus('imagenes/checker.bmp', 2.2, 11).save('resultados/checker_gaus_11.png')
    # gaus('imagenes/checker.bmp', 2.6, 13).save('resultados/checker_gaus_13.png')
    # gaus('imagenes/checker.bmp', 3, 15).save('resultados/checker_gaus_15.png')
    # gaus('imagenes/checker.bmp', 2.4, 17).save('resultados/checker_gaus_17.png')
    # gaus('imagenes/checker.bmp', 3.8, 19).save('resultados/checker_gaus_19.png')
    # gaus('imagenes/checker.bmp', 4.2, 21).save('resultados/checker_gaus_21.png')


    # mediana('imagenes/checker.bmp', 2).save('resultados/checker_mediana2.png')
    # mediana('imagenes/checker.bmp', 3).save('resultados/checker_mediana3.png')
    # mediana('imagenes/checker.bmp', 5).save('resultados/checker_mediana5.png')
    # mediana('imagenes/checker.bmp', 7).save('resultados/checker_mediana7.png')
    # mediana('imagenes/checker.bmp', 9).save('resultados/checker_mediana9.png')
    # mediana('imagenes/checker.bmp', 11).save('resultados/checker_mediana11.png')
    #
    # mediana('imagenes/escimp5.bmp', 2).save('resultados/escimp5_mediana2.png')
    # mediana('imagenes/escimp5.bmp', 3).save('resultados/escimp5_mediana3.png')
    # mediana('imagenes/escimp5.bmp', 5).save('resultados/escimp5_mediana5.png')
    # mediana('imagenes/escimp5.bmp', 7).save('resultados/escimp5_mediana7.png')
    # mediana('imagenes/escimp5.bmp', 9).save('resultados/escimp5_mediana9.png')
    # mediana('imagenes/escimp5.bmp', 11).save('resultados/escimp5_mediana11.png')
    #
    # mediana('imagenes/escgaus.bmp', 2).save('resultados/escgaus_mediana2.png')
    # mediana('imagenes/escgaus.bmp', 3).save('resultados/escgaus_mediana3.png')
    # mediana('imagenes/escgaus.bmp', 5).save('resultados/escgaus_mediana5.png')
    # mediana('imagenes/escgaus.bmp', 7).save('resultados/escgaus_mediana7.png')
    # mediana('imagenes/escgaus.bmp', 9).save('resultados/escgaus_mediana9.png')
    # mediana('imagenes/escgaus.bmp', 11).save('resultados/escgaus_mediana11.png')

    #
    # bilateral('imagenes/checker.bmp', 9, 300, 75).save('resultados/checker_bilateral75.png')
    # bilateral('resultados/checker_bilateral75.png', 9, 300, 75).save('resultados/checker_bilateral75_2.png')
    # bilateral('resultados/checker_bilateral75_2.png', 9, 300, 75).save('resultados/checker_bilateral75_3.png')
    # bilateral('imagenes/checker.bmp', 9, 110, 110).save('resultados/checker_bilateral110.png')
    # bilateral('resultados/checker_bilateral75.png', 9, 110, 110).save('resultados/checker_bilateral110_2.png')
    # bilateral('resultados/checker_bilateral75_2.png', 9, 110, 110).save('resultados/checker_bilateral110_3.png')
    #
    # bilateral('imagenes/escimp5.bmp', 9, 75, 75).save('resultados/escimp5_bilateral75.png')
    # bilateral('resultados/escimp5_bilateral75.png', 9, 75, 75).save('resultados/escimp5_bilateral75_2.png')
    # bilateral('resultados/escimp5_bilateral75_2.png', 9, 75, 75).save('resultados/escimp5_bilateral75_3.png')
    # bilateral('imagenes/escimp5.bmp', 9, 110, 110).save('resultados/escimp5_bilateral110.png')
    # bilateral('resultados/escimp5_bilateral75.png', 9, 110, 110).save('resultados/escimp5_bilateral110_2.png')
    # bilateral('resultados/escimp5_bilateral75_2.png', 9, 110, 110).save('resultados/escimp5_bilateral110_3.png')
    #
    # bilateral('imagenes/escimp5.bmp', 9, 50, 50).save('resultados/escimp5_bilateral50.png')
    # bilateral('resultados/escimp5_bilateral75.png', 9, 50, 50).save('resultados/escimp5_bilateral50_2.png')
    # bilateral('resultados/escimp5_bilateral75_2.png', 9, 50, 50).save('resultados/escimp5_bilateral50_3.png')
    #
    #
    # bilateral('imagenes/escgaus.bmp', 9, 75, 75).save('resultados/escgaus_bilateral75.png')
    # bilateral('resultados/escgaus_bilateral75.png', 9, 75, 75).save('resultados/escgaus_bilateral75_2.png')
    # bilateral('resultados/escgaus_bilateral75_2.png', 9, 75, 75).save('resultados/escgaus_bilateral75_3.png')
    # bilateral('imagenes/escgaus.bmp', 9, 110, 110).save('resultados/escgaus_bilateral110.png')
    # bilateral('resultados/escgaus_bilateral75.png', 9, 110, 110).save('resultados/escgaus_bilateral110_2.png')
    # bilateral('resultados/escgaus_bilateral75_2.png', 9, 110, 110).save('resultados/escgaus_bilateral110_3.png')






main();
