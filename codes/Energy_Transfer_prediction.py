#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:34:45 2022

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm

c = 1
f = 1
f0 = 1e-4


def Gamma(k, theta):
    omega = (f**2 + c**2*k**2)**0.5
    A = c*k**2/4/omega**2
    denom = (f**2 + 4*c**2*k**2*np.sin(theta/2)**2)**0.5
    
    real = (f**2 + omega**2)*np.sin(2*theta) - f**2*np.sin(2*theta)
    imag = f*omega*(2*np.cos(2*theta) - np.cos(theta))
    return A*(real + 1j*imag)/denom

eps = 1
k =1
A = 1
p1_0 = 0
p3_0 = A
q2_0 = eps

angle  = 60*np.pi/180

natural_frequency = 2*np.abs(Gamma(k, angle ))*q2_0

complex_oscillation = -2*Gamma(k, angle )*q2_0/natural_frequency
print (complex_oscillation)


delta_t = np.pi/(2*eps*natural_frequency)

t = np.linspace(0, 1, 1000)*delta_t

p1 = A*complex_oscillation*np.sin(natural_frequency*t)
#p4 = A*np.sin(natural_frequency*t)
p3 = A*np.cos(natural_frequency*t)

plt.plot(t, np.abs(p1), label = 'p1')
plt.plot(t, np.abs(p3), label = 'p3')
plt.plot(t[0:-500], np.abs(p1[0:-500])/np.abs(p3[0:-500]))

plt.legend()
plt.show()





#
#def func(x, y):
#    return 1./(1 +x*np.sin(y)**2)**(0.5)
#
#
#y = np.linspace(-180, 180, 1000)*np.pi/180
#x1 = .9
#x2 = 1.1
#
#plt.plot(y, func(x1, y))
#plt.plot(y, func(x2, y))
#plt.show()
