#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:08:11 2021

@author: jeff
"""


from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from scipy.signal import argrelextrema, peak_widths, find_peaks

t = np.linspace(-1, 1, 101)

g = np.exp(-t**2) + np.random.random(101)*0.11

#plt.plot(t, g)
#plt.show()

print (np.where(g==np.max(g)))


x = np.random.random(101)
s = np.sin(t*9) + x

#
#print (argrelextrema(s, np.greater))

def region_max(data, domain):
    return np.max(data[domain[0], domain[-1]])
#
#



Lr = np.arange(1, 5, 0.5)

def angle(Lr):
    theta  = np.arcsin(1./Lr/2)
    return theta # np.rad2deg(theta)

#
#theta = angle(Lr)
#print (theta)
#plt.plot(Lr, theta)
#plt.show()
    
##peaks, _ = find_peaks(g)
##print (peaks, 'peaks')
#results_half = peak_widths(g, np.array([50]), rel_height = 0.5)
##print (results_half)
#
#dt = t[1] - t[0]
#print (results_half[0][0]*dt)
#
#plt.plot(t, g)
#plt.show()
#
#plt.plot( g)
#plt.show()


def closest_ind(array, value):
    index, val =  min (enumerate(array), key=lambda x: abs(x[1]-value))
    return index

def FWHM(x, t):
    half_max = np.max(x)/2

    max_ind = np.where(np.max(x)==x)[0][0]

    left = x[:max_ind]
    right = x[max_ind:]
    t_left = t[:max_ind]
    t_right = t[max_ind:]
    ind_right = closest_ind(right, half_max)
    ind_left = closest_ind(left, half_max)

    width = t_right[ind_right] -t_left[ind_left]
    
    return width

print (FWHM(g, t))
plt.plot(t, g)

plt.show()

plt.plot( g)
plt.show()



