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



def caja(n, m, imgName, newImgName):
    img = imread(imgName, True);
    result = np.ones((n,m))
    result =  result /(n*m)
    result = ndimage.convolve(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save(newImgName)

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
    # res = np.ones(img.size)
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



def ex(x, n, sigma):
    return math.exp( (-1/2) * ( (x-((n-1)/2))/ sigma)**2 )
## Filtro gausiano
def gaus(sigma_1, sigma_2, n, m, imgName, newImgName):
    img = imread(imgName, True)
    ac = 0
    rx = np.ones((1,n));
    ry = np.ones((m, 1));


    for i in range(0, n):
        ac += ex(i, n, sigma_1)
        rx[0][i] = ex(i, n, sigma_1) * ac**(-1)

    ac = 0
    for i in range(0, m):
        ac += ex(i, m, sigma_2)
        ry[i][0] = ex(i, m, sigma_2) * ac**(-1)

    t = rx * ry;
    t = t/ np.sum(t)

    result = ndimage.convolve(img, t, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save(newImgName)



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

# if __name__=='__main__':
#     from timeit import Timer
#     t1 = Timer("caja(3,3)", "from __main__ import caja")
#     t2 = Timer("caja_convolve1(3,3)", "from __main__ import caja_convolve1")
#     print(t1.timeit(1), "caja")
#     print(t2.timeit(1), "1d")


def main():
    bilateral('imagenes/checker.bmp', 2, 40, 40).save('resultados/checker_bilateral2.bmp')
    bilateral('imagenes/checker.bmp', 3, 40, 40).save('resultados/checker_bilateral3.bmp')
    bilateral('imagenes/checker.bmp', 5, 40, 40).save('resultados/checker_bilateral5.bmp')
    bilateral('imagenes/checker.bmp', 7, 40, 40).save('resultados/checker_bilateral7.bmp')
    bilateral('imagenes/checker.bmp', 9, 40, 40).save('resultados/checker_bilateral9.bmp')

    bilateral('imagenes/escimp5.bmp', 2, 40, 40).save('resultados/escimp5_bilateral2.bmp')
    bilateral('imagenes/escimp5.bmp', 3, 40, 40).save('resultados/escimp5_bilateral3.bmp')
    bilateral('imagenes/escimp5.bmp', 5, 40, 40).save('resultados/escimp5_bilateral5.bmp')
    bilateral('imagenes/escimp5.bmp', 7, 40, 40).save('resultados/escimp5_bilateral7.bmp')
    bilateral('imagenes/escimp5.bmp', 9, 40, 40).save('resultados/escimp5_bilateral9.bmp')

    bilateral('imagenes/escgaus.bmp', 2, 40, 40).save('resultados/escgaus_bilateral2.bmp')
    bilateral('imagenes/escgaus.bmp', 3, 40, 40).save('resultados/escgaus_bilateral3.bmp')
    bilateral('imagenes/escgaus.bmp', 5, 40, 40).save('resultados/escgaus_bilateral5.bmp')
    bilateral('imagenes/escgaus.bmp', 7, 40, 40).save('resultados/escgaus_bilateral7.bmp')
    bilateral('imagenes/escgaus.bmp', 9, 40, 40).save('resultados/escgaus_bilateral9.bmp')



    # mediana('imagenes/checker.bmp', 2).save('resultados/checker_mediana2.bmp')
    # mediana('imagenes/checker.bmp', 3).save('resultados/checker_mediana3.bmp')
    # mediana('imagenes/checker.bmp', 5).save('resultados/checker_mediana5.bmp')
    # mediana('imagenes/checker.bmp', 7).save('resultados/checker_mediana7.bmp')
    # mediana('imagenes/checker.bmp', 9).save('resultados/checker_mediana9.bmp')
    #
    # mediana('imagenes/escimp5.bmp', 2).save('resultados/escimp5_mediana2.bmp')
    # mediana('imagenes/escimp5.bmp', 3).save('resultados/escimp5_mediana3.bmp')
    # mediana('imagenes/escimp5.bmp', 5).save('resultados/escimp5_mediana5.bmp')
    # mediana('imagenes/escimp5.bmp', 7).save('resultados/escimp5_mediana7.bmp')
    # mediana('imagenes/escimp5.bmp', 9).save('resultados/escimp5_mediana9.bmp')
    #
    # mediana('imagenes/escgaus.bmp', 2).save('resultados/escgaus_mediana2.bmp')
    # mediana('imagenes/escgaus.bmp', 3).save('resultados/escgaus_mediana3.bmp')
    # mediana('imagenes/escgaus.bmp', 5).save('resultados/escgaus_mediana5.bmp')
    # mediana('imagenes/escgaus.bmp', 7).save('resultados/escgaus_mediana7.bmp')
    # mediana('imagenes/escgaus.bmp', 9).save('resultados/escgaus_mediana9.bmp')


    # img1 = imread('imagenes/checker.bmp', True);
    # gaus(6,6,30,30, 'imagenes/checker.bmp', 'resultados/checker_gaus30.bmp');
    # caja(30,30, 'imagenes/checker.bmp', 'resultados/checker_caja30.bmp');
    # resultado = Image.fromarray(ndimage.gaussian_filter(img1, 6).astype(np.uint8))
    # resultado.save('resultados/cheker_gausiana30.bmp')
    # gaus(1, 1, 5, 5, 'imagenes/escgaus.bmp', 'resultados/escgaus_gaus_5.bmp');
    # gaus(1.4, 1.4, 7, 7, 'imagenes/escgaus.bmp', 'resultados/escgaus_gaus_7.bmp');
    # gaus(1.8, 1.8, 9, 9, 'imagenes/escgaus.bmp', 'resultados/escgaus_gaus_9.bmp');
    # gaus(2.2, 2.2, 11, 11, 'imagenes/escgaus.bmp', 'resultados/escgaus_gaus_11.bmp');
    # gaus(0.6, 0.6, 3,3, 'imagenes/escimp5.bmp', 'resultados/escimp5_gaus_3.bmp');
    # gaus(1, 1, 5, 5, 'imagenes/escimp5.bmp', 'resultados/escimp5_gaus_5.bmp');
    # gaus(1.4, 1.4, 7, 7, 'imagenes/escimp5.bmp', 'resultados/escimp5_gaus_7.bmp');
    # gaus(1.8, 1.8, 9, 9, 'imagenes/escimp5.bmp', 'resultados/escimp5_gaus_9.bmp');
    # gaus(2.2, 2.2, 11, 11, 'imagenes/escimp5.bmp', 'resultados/escimp5_gaus_11.bmp');
    # caja(3,3, 'imagenes/escgaus.bmp', 'resultados/escgaus_caja_3.bmp');
    # caja(5, 5, 'imagenes/escgaus.bmp', 'resultados/escgaus_caja_5.bmp');
    # caja(7, 7, 'imagenes/escgaus.bmp', 'resultados/escgaus_caja_7.bmp');
    # caja(9, 9, 'imagenes/escgaus.bmp', 'resultados/escgaus_caja_9.bmp');
    # caja(11, 11, 'imagenes/escgaus.bmp', 'resultados/escgaus_caja_11.bmp');
    # caja(3,3, 'imagenes/escimp5.bmp', 'resultados/escimp5_caja_3.bmp');
    # caja(5, 5, 'imagenes/escimp5.bmp', 'resultados/escimp5_caja_5.bmp');
    # caja(7, 7, 'imagenes/escimp5.bmp', 'resultados/escimp5_caja_7.bmp');
    # caja(9, 9, 'imagenes/escimp5.bmp', 'resultados/escimp5_caja_9.bmp');
    # caja(11, 11, 'imagenes/escimp5.bmp', 'resultados/escimp5_caja_11.bmp');
    # img1 = imread('imagenes/escimp5.bmp', True);
    # img2 = imread('imagenes/escgaus.bmp', True);
    # resultado = Image.fromarray(ndimage.gaussian_filter(img1, 3).astype(np.uint8))
    # resultado.save('resultados/gaus1.bmp')
    # resultado = Image.fromarray(ndimage.gaussian_filter(img1, 5).astype(np.uint8))
    # resultado.save('resultados/gaus5.bmp')
    # resultado = Image.fromarray(ndimage.gaussian_filter(img1, 7).astype(np.uint8))
    # resultado.save('resultados/gaus7.bmp')
    # resultado = Image.fromarray(ndimage.gaussian_filter(img1, 9).astype(np.uint8))
    # resultado.save('resultados/gaus9.bmp')
    # resultado = Image.fromarray(ndimage.gaussian_filter(img1,11).astype(np.uint8))
    # resultado.save('resultados/gaus11.bmp')
    #
    #
    #

    # resultado = Image.fromarray(ndimage.gaussian_filter(img2, 5).astype(np.uint8))
    # resultado.save('resultados/gausx5.bmp')
    # resultado = Image.fromarray(ndimage.gaussian_filter(img2, 7).astype(np.uint8))
    # resultado.save('resultados/gausx7.bmp')
    # resultado = Image.fromarray(ndimage.gaussian_filter(img2, 9).astype(np.uint8))
    # resultado.save('resultados/gausx9.bmp')
    # resultado = Image.fromarray(ndimage.gaussian_filter(img2,11).astype(np.uint8))
    # resultado.save('resultados/gausx11.bmp')

main();
