#################################
# Segmentacion de imagen a la "Grab Cut" simplificado
# por Luis Baumela. UPM. 15-10-2015
# Vision por Computador. Mastefr en Inteligencia Artificial
#################################


import numpy as np
import scipy
from scipy.misc import imread
import maxflow
import matplotlib.pyplot as plt
import select_fg_bg_pixels as sel
import cv2
import math

imgName='imagenes/horse.jpg'

img = imread(imgName)

im = imread(imgName)
# Marco el objeto y el fondo
markedImgName = sel.select_fg_bg(imgName, img.shape,3)

# Create the graph.
g = maxflow.Graph[float]()

# Add the nodes. nodeids has the identifiers of the nodes in the grid.
# height 375
# width 500
# img.shape[:2]  (375,500)
nodeids = g.add_grid_nodes(img.shape[:2])

# Calcula los costes de los nodos no terminales del grafo

# Estos son los costes de los vecinos horizontales

#for i in range
def aux(ip, iq, sigma):
    return math.exp(-((np.linalg.norm(ip, 2)-np.linalg.norm(iq, 2))**2)/2*sigma**2)

h, w =img.shape[:2]
# Estos son los costes de los vecinos horizontales
exp_aff_h = np.ones(img.shape[:2])
# Estos son los costes de los vecinos verticales
exp_aff_v = np.ones(img.shape[:2])
for i in range(0, h):
    for j in range(1, w):
        exp_aff_h[i, j] = aux(img[i][j], img[i][j-1], 2.3)

for i in range(1, h):
    for j in range(0, w):
        exp_aff_v[i, j] = aux(img[i][j], img[i-1][j], 2.3)
# Construyo el grafo
hor_struc=np.array([[0, 0, 0],[1, 0, 0],[0, 0, 0]])# mira vecino izquierdo
ver_struc=np.array([[0, 1, 0],[0, 0, 0],[0, 0, 0]])# mira vecino de arriba

g.add_grid_edges(nodeids, exp_aff_h, hor_struc,symmetric=True)
g.add_grid_edges(nodeids, exp_aff_v, ver_struc,symmetric=True)

# Busco los pixeles marcados en la imagen
markImg=imread(markedImgName)
# Los marcados en rojo representan el objeto
pts_fg = np.transpose(np.where(np.all(np.equal(markImg,(255,0,0)),2)))
# Los marcados en verde representan el fondo
pts_bg = np.transpose(np.where(np.all(np.equal(markImg,(0,255,0)),2)))

result = np.ones(img.shape[:2])
r = np.ones(img.shape[:2])
result = result * 50
r = r * 50


for i in range(0, len(pts_fg)):
    result[pts_fg[i][0]][pts_fg[i][1]] = 0
    r[pts_fg[i][0]][pts_fg[i][1]] = np.inf

for i in range(0, len(pts_bg)):
    result[pts_bg[i][0]][pts_bg[i][1]] = np.inf
    r[pts_bg[i][0]][pts_bg[i][1]] = 0

# Pesos de los nodos terminales

g.add_grid_tedges(nodeids, r, result)
# g.add_grid_tedges(nodeids[:,  0], np.inf, 0)
# # Sink node connected to rightmost non-terminal nodes.
# g.add_grid_tedges(nodeids[:,  -1], 0, np.inf)
# g.add_grid_tedges(nodeids, pts_bg, 255-pts_bg)

# g.add_grid_tedges(nodeids, im, 255-im)
# g.add_grid_tedges(nodeids, np.inf,0)
#
# left = nodeids[:, 0]
# print left
# g.add_grid_tedges(left, np.inf, 0)
# right = nodeids[:, -1]
# g.add_grid_tedges(right, 0, np.inf)


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
