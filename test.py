import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import Image
import cv2


#img = cv2.imread('escilum.tif',0)
#print img
#equ = cv2.equalizeHist(img)
#print equ
#res = np.hstack((img,equ)) #stacking images side-by-side
#print res
#cv2.imwrite('res.png',res)



from PIL import Image

myIm = Image.open("escilum.tif")

myImResta = myIm.point(lambda x: x-100)
print lambda x: x-100
myImSuma = myIm.point(lambda x: x+100)

myImResta.save("contraste-100.png")
myImSuma.save("contraste+100.png")
