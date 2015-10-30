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
    result = np.ones((n,n))
    result =  result /(n*n)
    result = ndimage.convolve(img, result, mode='constant', cval=1.0)
    resultado = Image.fromarray(result.astype(np.uint8))
    resultado.save('resultados/t.bmp')


caja(5,5)
