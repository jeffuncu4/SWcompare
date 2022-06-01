#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:58:46 2022

@author: jeff


Filter incoming mode, then calculate flux, then integrate over semi circle 
boundary
"""


from simulationClass import Simulation 
import matplotlib.pyplot as plt
import numpy.fft as fft
import itertools as it
from matplotlib import cm
import numpy as np

from calculate_at_boundary import path_integral

ro = 0.02
bu = 1.0
lr = 3.
ur = 1000.

exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
exp = Simulation(ro, bu, lr, ur)
exp.run_sim()
ana = exp.analysis
h = ana.h[-1] - ana.h[0]

print (np.shape(ana.h))
h_fft = fft.fftshift(fft.rfft2(h), axes = (0,1))

plt.imshow(np.abs(h_fft))
plt.colorbar()
plt.show()

h_fft[256, :]  =0.


h_filt = fft.irfft2(fft.ifftshift(h_fft, axes = (0,1)))


Lx = exp.Lx
ld = exp.Ld
dim  = Lx/ld

extent = [-dim,dim, -dim,dim]

normalize = np.max(ana.h[-1, 20, 20:100]) - ana.H
print (normalize)
#plt.imshow(h_filt/normalize, cmap = 'bwr', extent = extent)
#plt.xlabel(r'$x/L_d$', fontsize = 'x-large')
#plt.ylabel(r'$y/L_d$', fontsize = 'x-large')
#clb = plt.colorbar()
#clb.ax.set_title(r'$\eta/h_w$', fontsize = 'x-large')
#plt.title('Scattered Wave Height', fontsize = 'x-large')
#plt.show()

def remove_incoming(data):
    axes = (1, 2)
    data_fft  = fft.fftshift(fft.rfft2(data), axes = axes)
    data_fft[:, 256, :]  =0.
    return fft.irfft2(fft.ifftshift(data_fft, axes = axes))

start = 300
h_no_vortex = ana.h[start:] - ana.h[0]
u_no_vortex = ana.u[start:] - ana.u[0]
v_no_vortex = ana.v[start:] - ana.v[0]

h_filt = remove_incoming(h_no_vortex)
u_filt = remove_incoming(u_no_vortex)
v_filt = remove_incoming(v_no_vortex)

#plt.imshow(ana.u[-1] - ana.u[0])
#plt.title('u_filt')
#plt.show()

u_last = ana.u[start:] - ana.u[0]

#plt.plot(u_last[:, 10, 10])
#plt.show()
#plt.imshow(u_filt/normalize, cmap = 'bwr', extent = extent)
#plt.xlabel(r'$x/L_d$', fontsize = 'x-large')
#plt.ylabel(r'$y/L_d$', fontsize = 'x-large')
#clb = plt.colorbar()
#clb.ax.set_title(r'$\eta/h_w$', fontsize = 'x-large')
#plt.title('Scattered Wave Height', fontsize = 'x-large')
#plt.show()
#

def flux(h, u, v):
    u_squared = u**2 + v**2
    mult = exp.g*h**2  + h*u_squared/2 
    Fx = u*mult
    Fy = v*mult
    Fm = np.sqrt(Fx**2 + Fy**2)
    return Fx, Fy, Fm

Fx, Fy, Fm = flux(h_filt, u_filt, v_filt)

#Fx, Fy, Fm = flux(h_no_vortex, u_no_vortex, v_no_vortex)
#


print (np.min(Fm), np.min(Fm))
plt.imshow(Fm[-1])

plt.show()


Fm_ave = np.sum(Fm[0:8], axis = 0)




nx = 512

x = np.arange(nx)
mid = nx//2 -0.5
rad = 50

path_x = np.array([mid - rad, mid - rad])
path_y = np.array([mid - rad, mid + rad])

theta = np.linspace(-90, 90, 100)*np.pi/180
circle_x = mid + rad*np.cos(theta)
circle_y = mid + rad*np.sin(theta)

flux_before = path_integral(Fm_ave, x, x, path_x, path_y)/(2*rad)
flux_after = path_integral(Fm_ave, x, x, circle_x, circle_y)/(np.pi*rad)

print ('flux_before  = {}, flux_after = {}'.format(flux_before, flux_after))

plt.imshow(Fm_ave)
plt.scatter(mid, mid)
plt.scatter(path_x, path_y)
plt.scatter(circle_x, circle_y)
plt.colorbar()
plt.show()


























