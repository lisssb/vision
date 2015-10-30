#################################
# Segmentacion de imagen a la "Grab Cut" simplificado
# por Luis Baumela. UPM. 15-10-2015
# Vision por Computador. Master en Inteligencia Artificial
#################################


import numpy as np
import scipy
from scipy.misc import imread
import maxflow
import matplotlib.pyplot as plt
import select_fg_bg_pixels as sel

imgName='imgs/horse.jpg'

img = imread(imgName)

# Marco el objeto y el fondo
markedImgName = sel.select_fg_bg(imgName, img.shape,3)


# Create the graph.
g = maxflow.Graph[float]()

# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape[:2])

# Calcula los costes de los nodos no terminales del grafo

# Estos son los costes de los vecinos horizontales
exp_aff_h=
# Estos son los costes de los vecinos verticales
exp_aff_v=


# Construyo el grafo
hor_struc=np.array([[0, 0, 0],[1, 0, 0],[0, 0, 0]])
ver_struc=np.array([[0, 1, 0],[0, 0, 0],[0, 0, 0]])

g.add_grid_edges(nodeids, exp_aff_h, hor_struc,symmetric=True)
g.add_grid_edges(nodeids, exp_aff_v, ver_struc,symmetric=True)

# Busco los pixeles marcados en la imagen
markImg=imread(markedImgName)
# Los marcados en rojo representan el objeto
pts_fg = np.transpose(np.where(np.all(np.equal(markImg,(255,0,0)),2)))
# Los marcados en verde representan el fondo
pts_bg = np.transpose(np.where(np.all(np.equal(markImg,(0,255,0)),2)))

# Pesos de los nodos terminales
g.add_grid_tedges(...)
g.add_grid_tedges(...)

# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)

plt.figure()
plt.imshow(np.uint8(np.logical_not(sgm)))

# Pesos para mostrar mas oscuro el fondo
wgs=(np.float_(np.logical_not(sgm))+0.3)/1.3

# Replico los pesos para cada canal y ordeno los indices
wgs=np.rollaxis(np.tile(wgs,(3,1,1)),0,3)
plt.figure()
# Mutiplico los canales por los pesos y muestro las imagenes
plt.imshow(np.uint8(np.multiply(img,wgs)))
plt.show()
