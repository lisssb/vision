import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import Image, ImageOps
import cv2
import operator


#img = cv2.imread('escilum.tif', 0)
#print img
#hist = cv2.calcHist([img],[0],None,[256],[0,256])
#print hist

#imagen = Image.open('escilum.tif')
#ecualizada = ImageOps.equalize(imagen)
#ecualizada.show()
#print np.asarray(ecualizada).flatten()
#plt.hist(np.asarray(ecualizada).flatten(), bins=255, color='yellow', range=(0, 255)) #range(0, 255)
#plt.show()

#ecualizada.save("ecualizada.jpg")


img = cv2.imread('escilum.tif',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
print 'eq', equ
cv2.imwrite('res.png',res)



#from PIL import Image
#myIm = Image.open("escilum.tif")

#myImResta = myIm.point(lambda x: x-100)
#print lambda x: x-100
#myImSuma = myIm.point(lambda x: x+100)

#myImResta.save("contraste-100.png")
#myImSuma.save("contraste+100.png")
