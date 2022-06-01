#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:10:52 2022

@author: jeff
"""

import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt


L = 1
nx = 100
x = np.linspace(0, L, nx)

T = 1
nt = 1000
t = np.linspace(0, T, nt)

omega1 = 2*np.pi/T*5
kx1 = 2*np.pi/L*5
ky1 = 2*np.pi/L*0

omega2 = 2*np.pi/T*2
kx2 = 2*np.pi/L*0
ky2 = 2*np.pi/L*1

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
    

z =   np.cos(kx2*xxx + ky2*yyy - omega2*ttt) + np.sin(kx1*xxx + ky1*yyy - omega1*ttt) 





#print (np.shape(z[0]))
#
#plt.imshow(z[0])
#plt.colorbar()
#plt.show()
#
#plt.plot(t, z[:, 0, 0])
#plt.show()

z_ft = fft.fftshift(fft.fft(z, axis = 0)) *(t[1] - t[0])
omega = fft.fftshift(fft.fftfreq(nt, d = (t[1] - t[0])))

print (np.shape(omega))

#plt.plot(omega, np.abs(z_ft[:, 0, 0]))
#plt.title('z_ft ')
#plt.show()

z_ft[:nt//2, :, :] = 0.0
z_ft = z_ft*2

#plt.plot(omega, np.abs(z_ft[:, 0, 0]))
#plt.title('z_ft ')
#plt.show()

z_ft_st = fft.fftshift(fft.fft2(z_ft, axes = (1, 2)), axes = (1,2))*(x[1]-x[0])**2
k = fft.fftshift(fft.fftfreq(nx, d = (x[1] - x[0])))
l = fft.fftshift(fft.fftfreq(nx, d = (x[1] - x[0])))


omega1_ind, closest_val = min(enumerate(omega), key=lambda x: abs(x[1] + omega1)) # the plus is to account for -omega1*t

print (omega1_ind, closest_val, omega1)


print (np.max(z_ft_st))

extent = [k[0], k[-1], k[0], k[-1]]
#
#plt.imshow(np.abs(z_ft_st[505, :, :]), extent =extent)
#plt.title('omega1 wavennumber space')
#plt.colorbar()
#plt.show()
#
#
#plt.plot( np.abs(z_ft_st[:, 50, 55]))
#plt.title('omega1 wavennumber space')
#
#plt.show()

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm
import numpy.fft as fft





Ro = 0.01
Bu = 1.0
Lr = 4.0
Ur = 1000.0


exp = Simulation(Ro, Bu, Lr, Ur)
exp.run_sim()
ana = exp.analysis
h = ana.h[501:, :, :]
print (np.shape(h), 'h shape')
h0 = ana.h[0, :, :]
hw = h - h0

print (np.shape(hw), 'hw shape')
#plt.imshow(hw[0])
#plt.colorbar()
#plt.show()

t = ana.time_axis
x = ana.x_axis
nx = ana.nx
nt = len(hw)


z_ft = fft.fftshift(fft.fft(hw, axis = 0)) *(t[1] - t[0])
omega = fft.fftshift(fft.fftfreq(nt, d = (t[1] - t[0])))

print (np.max(np.abs(z_ft)), 'before')

print (np.shape(z_ft), nt, 'shape and length of z_ft')
z_ft[:nt//2, :, :] = 0.0
z_ft = z_ft*2
print (np.max(np.abs(z_ft)))

z_ft_st = fft.fftshift(fft.fft2(z_ft, axes = (1, 2)), axes = (1,2))*(x[1]-x[0])**2
k = fft.fftshift(fft.fftfreq(nx, d = (x[1] - x[0])))
l = fft.fftshift(fft.fftfreq(nx, d = (x[1] - x[0])))


print (np.where(z_ft_st == np.max(z_ft_st)))
plt.imshow(np.abs(z_ft_st[99, :, :]))
plt.colorbar()
plt.show()

z_ft_st[:, 256, 215] = 0.
z_ft_st[:, 256, :] = 0.

plt.imshow(np.abs(z_ft_st[99, :, :]))
plt.colorbar()
plt.show()
#
#plt.plot(np.abs(z_ft_st[:, 256, 215]))
#plt.show()

def complex_demod_ifft(data_ft_st, t, x):
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    data_ft = fft.ifft2(fft.ifftshift(data_ft_st, axes = (1,2)), axes = (1, 2))/(dx)**2
    data = fft.ifft(fft.ifftshift(data_ft), axis = 0)/dt
    return data

data = complex_demod_ifft(z_ft_st, t, x)

print (np.max(np.abs(data)-np.abs(hw)))

plt.imshow(np.real(data[0]))
plt.colorbar()
plt.show()



def complex_demod_fft(data, x, t):
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    nt = len(t)
    
    data_ft = fft.fftshift(fft.fft(data, axis = 0)) *dt

    data_ft[:nt//2, :, :] = 0.0 #only for even arrays I think
    data_ft = data_ft*2


    data_ft_st = fft.fftshift(fft.fft2(data_ft, axes = (1, 2)), axes = (1,2))*dx**2
    
    return data_ft_st

