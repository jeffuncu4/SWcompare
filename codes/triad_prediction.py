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
#
#exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
#exp = Simulation(ro, bu, lr, ur)
#exp.run_sim()
#
#f = exp.f
#c = exp.c
#


f = 1
c = 1
k = 1


def Gamma(k, theta):
    omega = (f**2 + c**2*k**2)**0.5
    A = c*k**2/4/omega**2
    denom = (f**2 + 4*c**2*k**2*np.sin(theta/2)**2)**0.5
    
    real = (f**2 + omega**2)*np.sin(2*theta) - f**2*np.sin(theta)
    imag = f*omega*(2*np.cos(2*theta) - np.cos(theta))
    return A*(real + 1j*imag)/denom






    

theta = np.linspace(0, 180, 1000)*np.pi/180
#k = exp.wavelength 

plt.plot(theta/np.pi*180, np.abs(Gamma(k, theta)), label = 'abs')
#plt.plot(theta/np.pi*180, np.real(Gamma(k, theta)), label = 'real')
#plt.plot(theta/np.pi*180, np.imag(Gamma(k, theta)), label = 'imag')
plt.xlabel('angle')
plt.ylabel('Gamma Structure factor')
plt.legend()
plt.show()


k_array = np.array([1, 1.5, 2, 3, 4])
k_array = np.arange(1, 10, 0.2)
theta_max = []
for k in k_array:
    scat_strength = np.abs(Gamma(k, theta))

    max_ind = np.where(scat_strength == np.max(scat_strength))[0][0]
    theta_max.append(theta[max_ind]*180/np.pi)
#    print (theta_max)
#    plt.plot(theta*180/np.pi, scat_strength)
#    plt.show()

plt.scatter(k_array, np.array(theta_max))
plt.xlabel('k')
plt.ylabel('strongest triad resonance angle')
plt.show()

#A = 1
#p1_0 = 0
#p3_0 = A
#q2_0 = 1
#
#angle  = 35*np.pi/180
#
#natural_frequency = 2*np.abs(Gamma(1, angle ))*q2_0
#
#complex_oscillation = -2*Gamma(1, angle )*q2_0/natural_frequency
#
#eps = 0.1
#delta_t = np.pi/(2*eps*natural_frequency)
#
#t = np.linspace(0, 1, 1000)*delta_t
#
#p1 = A*complex_oscillation*np.sin(natural_frequency*t)
#p3 = A*np.cos(natural_frequency*t)
#
#plt.plot(t, np.abs(p1), label = 'p1')
#plt.plot(t, np.abs(p3), label = 'p3')
#plt.legend()
#plt.show()
#
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
#
#
#


#def isosceles_theta(Lr):
#    return 2*np.arcsin(1./(2*Lr))
#
#
#Lr = np.array([1,1.5,2,3,4])
#Lr = np.linspace(1, 5, 100)
#
#
#k2 = 1
#k1 = Lr*k2
#
#gl  = np.abs(Gamma(k1, isosceles_theta(Lr)))
#
#for l in Lr:
#    plt.plot(np.abs(Gamma(k1, isosceles_theta(l))))
#plt.show()
#print (gl)
#plt.plot(Lr, gl)
#plt.show()
##























