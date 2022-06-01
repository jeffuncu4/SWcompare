#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:58:35 2022

@author: jeff


This function will find indices in the range of rad1-rad2 and angles phi1-phi2

theta = 0 is the positive y axis
"""

from matplotlib import cm
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
#
a = 50
nx, ny  = a, a

h = np.ones([nx, ny])


rad1 = 5
rad2 = 7
phi1 = np.pi*0
phi2 = np.pi*1/2

def find_allowed_index(h, rad1, rad2, phi1, phi2):
    nx, ny = np.shape(h)
    indi = []
    indj = []
    r0 = nx//2 - 0.5
    
    for i in range(nx):
        it =  i - r0
        for j in range (ny):
            jt = j - r0
            r = (it**2 + jt**2)**0.5
            if (rad1 < r and r < rad2):
                
#                indi.append(i)
#                indj.append(j)
    
                angle = -np.arctan2(-it, jt) + np.pi
                if phi1 <= angle <= phi2:
                    indi.append(i)
                    indj.append(j)
    
    return indi, indj


indi, indj = find_allowed_index(h, rad1, rad2, phi1, phi2)

plt.imshow(h)
plt.scatter(indj, indi)
plt.scatter(nx//2-0.5, nx//2-0.5)
plt.show()





def filter_field(h, rad1, rad2, phi1, phi2):
    indi, indj = find_allowed_index(h, rad1, rad2, phi1, phi2)
    print (indi, indj)
    h_remainder = np.copy(h)
    h_extract = np.zeros(np.shape(h))#, dtype = np.complex128)
    
    h_extract[indi, indj] = h[indi, indj] 
    h_remainder[indi, indj]  = 0.
    return h_extract, h_remainder


#he, hr = filter_field(h, rad1, rad2, phi1, phi2)
#
#
#plt.imshow(he)
#plt.colorbar()
#plt.show()
#plt.imshow(hr)
#plt.colorbar()
#plt.show()


#
#from simulationClass import Simulation 
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#import itertools as it
#from matplotlib import cm
#import numpy.fft as fft
#
#
#
#
#
#Ro = 0.02
#Bu = 1.0
#Lr = 4.0
#Ur = 1000.0
#
#
#exp = Simulation(Ro, Bu, Lr, Ur)
#exp.run_sim()
#ana = exp.analysis
#h = ana.h[-1, :, :]
#h0 = ana.h[0, :, :]
#hw = h - h0
#
#plt.imshow(hw)
#plt.show()
#
#h_fft = (fft.fftshift(fft.fft2(hw)))
#
#plt.imshow(np.abs(h_fft))
#plt.title('orig')
#plt.colorbar()
#plt.show()
##
##ri = 297
#
#rad1 = ri - 258
#rad2 = ri  - 254
#
#phi1 = np.pi*0
#phi2 = np.pi
#
#he, hr = filter_field(h_fft, rad1, rad2, phi1, phi2)
#
#plt.imshow(np.abs(he))
#plt.title('h_extract')
#plt.colorbar()
#plt.show()
#plt.imshow(np.abs(hr))
#plt.title('h_remainder')
#plt.colorbar()
#plt.show()
#





