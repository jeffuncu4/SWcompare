#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:14:04 2021

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D

from itertools import product, combinations

#
#ro = 0.03
#bu = 1.
#lr = 2.
#ur = 1000.
#
#exp = Simulation(ro, bu, lr, ur)
#exp.run_sim()
##exp.plots.view('h')
#
#
#fx, fy, flux_diff = exp.analysis.flux_omega_averaged()
#skip = 14
#thin_fx = fx[::skip, ::skip]
#thin_fy = fy[::skip, ::skip]
#
#x,y = np.meshgrid(np.linspace(-1,1, 462//skip), np.linspace(-1,1, 462//skip))
#
#print (np.shape(x), np.shape(fx))
#fm = np.sqrt(fx**2 + fy **2)
#
#
#plt.subplot(1,2,1)
#plt.quiver(x,y, thin_fy, thin_fx)
#plt.imshow(fm, extent = [-1,1,-1,1], origin = 'lower')
#
#
#plt.subplot(1,2,2)
#vort = exp.analysis.vorticity()
#
#plt.imshow(vort, extent = [-1,1,-1,1], origin = 'lower')
#
#plt.colorbar()
#plt.show()
#






Ro  = [0.2]
Bu = [0.5, 0.9, 1.0, 1.1, 2]
Lr = [0, 4]


#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.set_aspect("auto")

## draw cube
##r = [-1, 1]
##for s, e in combinations(np.array(list(product(r, r, r))), 2):
##    if np.sum(np.abs(s-e)) == r[1]-r[0]:
##        ax.plot3D(*zip(s, e), color="b")
#
#
##
#ax.scatter([0], [0], [0], color="g", s=100)

#
#for i, j , k in product(Ro, Bu, Lr):
#    ax.scatter(i, j, k, color = 'g', s= 100)
#
#for i, j, k in product([0.2], [1], [1, 2, 3, 4]):
#    ax.scatter(i, j, k, color = 'b', s= 100)
#
#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")



Ro = [0.8, 1., 1.2 ]
Bu = [0.9, 1., 1.1]
Lr = [0.5, 1., 1.5]

for i, j , k in product(Ro, Bu, Lr):
    ax.scatter(i, j, k, color = 'g', s= 100)


Ro = [0.01, 0.1, 0.2]
Bu = [0.9, 1., 1.1]
Lr = [0.5, 1., 1.5]



for i, j, k in product(Ro, Bu, Lr):
    ax.scatter(i, j, k, color = 'b', s= 100)

ax.scatter(1, 1, 1, color = 'r', s = 100)
ax.scatter(0.1, 1, 1, color = 'r', s = 100)

ax.set_xlabel('Ro')
ax.set_xlim(0, 2)
ax.set_ylabel('Bu')
ax.set_ylim(0.5, 2)
ax.set_zlabel('Lr')
ax.set_zlim(0.1, 2)
plt.show()



fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")


Ro = [0., 0.1, 0.54, 1.3]
Lr = [2, 3]
Bu = [1.]

for i, j, k in product(Ro, Bu, Lr):
    ax.scatter(i, j, k, color = 'r', s= 100)

Ro = [0.1, 0.54]
Lr = [1, 2, 3, 4]
Bu = [1.]

for i, j, k in product(Ro, Bu, Lr):
    ax.scatter(i, j, k, color = 'b', s= 100)
    
Ro = [0.1, 0.54]
Lr = [2.]
Bu = [0.5, 0.9, 1., 1.1, 1.5]

for i, j, k in product(Ro, Bu, Lr):
    ax.scatter(i, j, k, color = 'g', s= 100)

ax.set_xlabel('Ro')

ax.set_ylabel('Bu')

ax.set_zlabel('Lr')

plt.show()

#Ro = [ 0.02]
#Bu = [0.5, 0.9,1., 1.1, 2]
#Lr = [2.]
#Ur = [1000.]
#
#
#
#
##
##plt.scatter(np.log(Bu), np.ones(len(Bu)))
##plt.show()
##
#
#ro = 0.01
#bu = 1.0
#lr = 2.0
#ur = 1000.0
#
#exp = Simulation(ro, bu, lr, ur)
#exp.run_sim()
##exp.plots.view('h')
#exp.analysis
##exp.analysis.flux_omega_averaged()
#fx, fy, flux_diff = exp.analysis.flux_omega_averaged()
##fx, fy = exp.analysis.flux_mag_averaged()
#Fm = np.sqrt(fx**2 +fy**2)
#plt.imshow(Fm)
#
#plt.colorbar()
#plt.show()
#
#
