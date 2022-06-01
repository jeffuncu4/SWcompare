#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:48:08 2022

@author: jeff

Ward and dewar triad resonoance prediction
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm

#ro = 0.03
#bu = 1.0
#lr = 4.0
#ur  = 1000.

#exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
#exp = Simulation(ro, bu, lr, ur)
#exp.run_sim()
#ana = exp.analysis 

#f = exp.f
#c = exp.c
#


L = 25e3

k_wave = np.pi*2/L
k_geo = np.pi*2/L



k = np.linspace(0.5*k_geo, k_geo*10, 500)
f = 1e-4
g = 9.81
c = f*L

#Lambda = k1/k2

theta = 2*np.arcsin(0.5*(k_wave/k))
plt.plot(k_wave/k, theta)
plt.show()

def Gamma(k, theta):
    omega = (f**2 + c**2*k**2)**0.5
    A = c*k**2/4/omega**2
    denom = (f**2 + 4*c**2*k**2*np.sin(theta/2)**2)**0.5
    
    real = (f**2 + omega**2)*np.sin(2*theta) - f**2*np.sin(theta)
    imag = f*omega*(2*np.cos(2*theta) - np.cos(theta))
    return np.abs(A*(real + 1j*imag)/denom)**2



def E(k, k2):
    a = 1
    return np.exp(-((k-k2)/a)**2)
#
#
#print (Gamma(k1, theta))

gamma = Gamma(k, theta)
plt.plot(k, gamma)
plt.show()
#gamma[0:34] =0.
w = gamma*E(k, k_geo)

plt.plot(k, E(k, k_geo)/ np.max(E(k, k_geo)), label = 'E')
plt.plot(k,gamma/np.max(gamma), label = 'gamma')
plt.plot(k, w/np.max(w), label = 'weight')
plt.xlabel('k')
plt.legend()
plt.show()


#plt.plot(Lambda, E(k, k2)/ np.max(E(k, k2)), label = 'E')
#plt.plot(Lambda,gamma/np.max(gamma), label = 'gamma')
#plt.plot(Lambda, w/np.max(w), label = 'weight')
#plt.xlabel('Lr')
#plt.legend()
#plt.show()





#def Gamma(k1, k2):
#    omega = (f**2 + c**2*k1**2)**0.5
#    theta = 2*np.arcsin(1/(2*k1/k2))
#    
#    A = c*k1**2/4/omega**2
#    denom = (f**2 + 4*c**2*k1**2*np.sin(theta/2)**2)**0.5
#    
#    real = (f**2 + omega**2)*np.sin(2*theta) - f**2*np.sin(theta)
#    imag = f*omega*(2*np.cos(2*theta) - np.cos(theta))
#    return np.abs(A*(real + 1j*imag)/denom)**2
#
#
#def len_scales(Lr):
#    k1  = 1
#    k2 = k1/Lr
#    return k1, k2
    
    
#Lr = np.array([1., 2., 3., 4.])
#
#k1, k2 = len_scales(Lr)
#print (k1, k2)
#
#
#g = Gamma(k1, k2)
#
#plt.plot(Lr, g)
#plt.show()
#

#plt.plot(Lambda, theta*180/np.pi)
#plt.title('Triad angles')
#plt.xlabel('Lr')
#plt.ylabel('angle')
#plt.show()



