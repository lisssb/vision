import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
img = mpimg.imread('escilum.tif')
plt.imshow(img, cmap=cm.Greys_r)
plt.show()
