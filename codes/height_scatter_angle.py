#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:05:12 2022

@author: jeff
"""


from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm


Ro = 0.02
Bu = 1.0
Lr = 4.0
Ur = 1000.0




exp = Simulation(Ro, Bu, Lr, Ur)
exp.run_sim()
ana = exp.analysis
h = ana.h[-1, :, :]
h0 = ana.h[0, :, :]

hw = h - h0
#
#plt.imshow(hw)
#plt.show()

import numpy.fft as fft

h_fft = np.abs(fft.fftshift(fft.fft2(hw)))*ana.dx

print (np.max(h_fft))
h_fft[256, :] = 0
#plt.imshow(h_fft[:, 256:])
#plt.show()

#
#a, b, c, d = ana.wavenumber_circle()
#
#print (a, b*180/np.pi, c*180/np.pi)

domain = 5*(ana.L)

nr = 1000
r = np.linspace(-domain, domain, nr)
sigma = ana.L/np.pi

gaussian = r*np.exp(-r**2/2/sigma**2)

plt.plot(r, gaussian)
plt.show()

gauss_fft = fft.fftshift(fft.fft(gaussian))#*np.exp(-1j*2*np.pi*)
freq = fft.fftshift(fft.fftfreq(len(r), d = (r[1]-r[0])))/(2*np.pi/ana.L)

plt.plot(freq, np.abs(gauss_fft))
plt.show()

theta_deg = np.linspace(-180, 180, 1000)

theta_rad = theta_deg*np.pi/180


a = np.linspace(0.5, 2)



k1 = 1.
N = 100.
k = np.arange(0, 2*k1, 2*k1/N)

def theta(k1, k):
    k2 = 2*np.arcsin(k/2/k1)
    return k2

plt.plot(k, theta(k1, k))
plt.show()





















