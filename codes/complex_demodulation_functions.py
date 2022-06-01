#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:32:07 2022

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm
import numpy.fft as fft

def complex_demod_fft(data, x, t):
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    nt = len(t)
    nx = len(x)
    
    omega = fft.fftshift(fft.fftfreq(nt, d = dt))
    k = fft.fftshift(fft.fftfreq(nx, d =dx))
    
    data_ft = fft.fftshift(fft.fft(data, axis = 0), axes = 0) *dt

    data_ft[:nt//2, :, :] = 0.0 #only for even arrays I think

    data_ft = data_ft*2
#    print (np.max(np.abs(data_ft)))
#    plt.plot(omega, np.abs(data_ft[:,0, 0]))
#    plt.show()


    data_ft_st = fft.fftshift(fft.fft2(data_ft, axes = (1, 2)), axes = (1, 2))*dx**2

#    print (np.max(np.abs(data_ft_st)))
#    plt.plot(omega, np.abs(data_ft[:,0, 0]))
#    plt.show()


    return data_ft_st


def complex_demod_ifft(data_ft_st, x, t):
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    data_ft = fft.ifft2(fft.ifftshift(data_ft_st, axes = (1, 2)), axes = (1, 2))/dx**2
    data = fft.ifft(fft.ifftshift(data_ft, axes = 0), axis = 0)/dt
    return np.real(data)


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


#indi, indj = find_allowed_index(h, rad1, rad2, phi1, phi2)
#
#plt.imshow(h)
#plt.scatter(indj, indi)
#plt.scatter(nx//2-0.5, nx//2-0.5)
#plt.show()
