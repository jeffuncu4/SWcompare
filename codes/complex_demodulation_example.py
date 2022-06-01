#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:34:40 2022

@author: jeff
"""


from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm
import numpy.fft as fft
from complex_demodulation_functions import complex_demod_fft, complex_demod_ifft



L = 1
nx = 100
x = np.linspace(0, L, nx)

T = 1
nt = 1000
t = np.linspace(0, T, nt)

omega1 = 2*np.pi/T*6
kx1 = 2*np.pi/L*0
ky1 = 2*np.pi/L*4

omega2 = 2*np.pi/T*20
kx2 = 2*np.pi/L*10
ky2 = 2*np.pi/L*10

xx , yy  = np.meshgrid(x, x)


xxx = np.zeros([nt, nx, nx])
yyy = np.zeros([nt, nx, nx])

for i in range(nt):
    xxx[i, :, :] = xx
    yyy[i, :, :] = yy

ttt = np.zeros([nt, nx, nx])
for i in range(nx):
    for j in range(nx):
        ttt[:, i, j] = t
    

z =   np.cos(kx2*xxx + ky2*yyy + omega2*ttt) + np.sin(kx1*xxx + ky1*yyy + omega1*ttt) 

#plt.imshow(z[0])
#plt.colorbar()
#plt.show()

z_ft_st = complex_demod_fft(z, x, t)
omega = fft.fftshift(fft.fftfreq(nt, d = (t[1] - t[0])))*np.pi*2/T
k = fft.fftshift(fft.fftfreq(nx, d = (x[1] - x[0])))


def find_ind_val(array, val):
    diff = np.abs(array - val)
    #print (diff)
    ind  = np.where(diff == np.min(diff))[0][0]
    return ind, array[ind]

omega1_ind, closest_val   = find_ind_val(omega, omega1)
omega2_ind, closest_val   = find_ind_val(omega, omega2)


extent = [k[0], k[-1], k[0], k[-1]]
print (k[0], k[-1])
plt.imshow(np.abs(z_ft_st[omega2_ind, :, :]), extent =extent)
plt.title('omega2 wavennumber space')
plt.colorbar()
plt.show()

extent = [k[0], k[-1], k[0], k[-1]]
plt.imshow(np.abs(z_ft_st[omega1_ind, :, :]), extent =extent)
plt.title('omega1 wavennumber space')
plt.colorbar()
plt.show()



z_cd = complex_demod_ifft(z_ft_st, x, t)

plt.imshow(z_cd[0])
plt.colorbar()
plt.show()

plt.imshow(z[0])
plt.colorbar()
plt.show()

plt.imshow((z_cd[0]- z[0])/np.max(z[0]))
plt.colorbar()
plt.show()











