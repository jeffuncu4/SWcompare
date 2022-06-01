#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:50:50 2022

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm
import numpy.fft as fft

a = 1
k1 = 1.*a
N = 1000.
k = np.arange(0, 2*k1, 2*k1/N)

def theta(k1, k):
    angle = 2*np.arcsin(k/2/k1)
    return angle

angle = theta(k1, k)
#plt.plot(k, theta(k1, k))
#plt.show()

def Gamma(a, theta):
    factor  = a/(1+a**2)/(1 + 4*a**2*np.sin(theta/2)**2)**0.5
    real_part = (2 + a**2)*np.sin(2*theta) - np.sin(theta)
    imag_part = (1 + a**2)**0.5*(2*np.cos(2*theta) - np.cos(theta))
    
    return factor*(real_part + 1j*imag_part)


gamma = Gamma(a, angle)

angle*=180/np.pi
plt.plot(k/k1, np.real(gamma), label  = 'real')
plt.plot(k/k1, np.imag(gamma), label = 'imag')
plt.plot(k/k1, np.abs(gamma), label = 'abs')
plt.legend()
plt.show()

#plt.plot(angle, np.real(gamma), label  = 'real')
#plt.plot(angle, np.imag(gamma), label = 'imag')
#plt.plot(angle, np.abs(gamma), label = 'abs')
#plt.legend()
#plt.show()


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



domain = 10*(ana.L)

nr = 1000*2
r = np.linspace(-domain, domain, nr)
dr = r[1] - r[0]
sigma = ana.L/np.pi

gaussian = r*np.exp(-r**2/2/sigma**2)

#plt.plot(r, gaussian)
#plt.show()


gauss_fft = fft.fftshift(fft.fft(gaussian))#*np.exp(-1j*2*np.pi*)
freq = fft.fftshift(fft.fftfreq(len(r), d = dr))/(np.pi/ana.L)*np.pi*2

plt.plot(freq[1000:], np.abs(gauss_fft[1000:])/np.max(np.abs(gauss_fft[1000:])))
plt.plot(k, np.abs(gamma)/np.max(np.abs(gamma)))
plt.show()




















