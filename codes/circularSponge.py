#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:10:32 2020

@author: jeff
"""


from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import h5py
from scipy import signal
import sys
import pickle

exp_dir = '../experiments/' + sys.argv[1] + '/'

par_dict = pickle.load(open(exp_dir + 'IC/parameters.pkl', 'rb'))

dx = par_dict['dx']
nx = par_dict['nx_v']
ny = par_dict['ny_v']
Lx = par_dict['Lx_v']
Ly = par_dict['Ly_v']
f = par_dict['f']
g = par_dict['g']
mu = par_dict['mu']
l = par_dict['l']
H = par_dict['H']
L = par_dict['L']
hg = par_dict['hg']



xbasis = de.Fourier('x', nx, interval=(-Lx/2,Lx/2), dealias=1) #dealias = 3/2 # this oworks with fourier(wihtout BC) and Chebyshev
ybasis = de.Fourier('y', ny, interval=(-Ly/2,Ly/2), dealias=1)

domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64)
x = domain.grid(0)
y = domain.grid(1)

h = domain.new_field(name='hg')


#slices = domain.dist.grid_layout.slices(scales=(1,1))

def gauss2d(x, y, sigx, sigy):
    return np.exp(-(x/sigx)**2/2-(y/sigy)**2/2)


h['g'] = hg*gauss2d(x, y, L/np.pi, L/np.pi)



cs = domain.new_field(name='circular sponge')


def circular_window(x, y, L):
    R1 = L*1.5
    R2=L*1.7
    window = np.ones([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xi = x[i]
            yi = y[0, j]
            r = xi**2 + yi**2
            if r < R1**2:
                window[i, j] = 0
            elif R1**2 <= r <= R2**2:
                window[i, j] = (r**0.5-R1)/(R2-R1)
                
    return window        
            

plot = True
if plot:
    cs['g'] = circular_window(x, y, L)
    plt.subplot(2,1,1)
    plt.imshow(cs['g'])
    plt.colorbar()
    
    plt.subplot(2,1,2)
    plt.imshow(h['g'])
    plt.colorbar()
    plt.show()

#csarray = cs['g'][nx//2, :]
#
#plt.plot(csarray)
#plt.plot(h['g'][nx//2, :])
#plt.show()
#
#

saveData=True
if saveData:
    hf = h5py.File(exp_dir + 'VIC/circularWindow.h5', 'w')
    hf.create_dataset('circularWindow', data=cs['g'])
    hf.close()


#T = np.pi*2/f
#t = np.linspace(-3*T, 3*T, 100)
#
#
#
#
#func = np.piecewise(t, [t < -2*T,  (t < -1*T ) & (t>-2*T) , t >= -1*T], [lambda t: 1, lambda t: (1 - (t+2*T)/T), lambda t: 0])
#print(func)
#
#
#plt.plot(func)
#plt.show()
#
#
#T = np.pi*2/f
#t = np.linspace(0, 1*T, 100)
#
#ct1 = T*0.2
#ct2 = T*0.3
#
#circularSpongeTimeRamp =  np.piecewise(t, [t < ct1,  (t < ct2 ) & (t>ct1) , t >= ct2], 
#                                           [lambda t: 1, lambda t: (1 - (t-ct1)/(ct2-ct1)), lambda t: 0])
#
#
#saveData=True
#if saveData:
#    hf = h5py.File('circularWindow.h5', 'w')
##    hf.create_dataset('circularWindow', data=cs['g'])
##    hf.close()
#T = np.pi*2/f
#t = np.linspace(0, 3*T, 100)
##
#f  = (1 - 1./(1 + np.exp(-20./T*(t -T))))
#
#plt.plot(t/T, f)
#plt.show()
#
























