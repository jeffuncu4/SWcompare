#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:04:28 2021

@author: jeff
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#x_ = np.linspace(0., 1., 10)
#y_ = np.linspace(1., 2., 20)
#z_ = np.linspace(3., 4., 30)
#
#x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
#
#assert np.all(x[:,0,0] == x_)
#assert np.all(y[0,:,0] == y_)
#assert np.all(z[0,0,:] == z_)

nx = 10

x, y, z = np.linspace(-1, 1, nx*2), np.linspace(-1, 1, nx), np.linspace(-1, 1, nx*3)

xx, yy, zz = np.meshgrid (x, y, z, indexing = 'ij')


assert np.all(xx[:,0,0] == x)

def fit_guess(x, y, z, a, b):
    return a*x**2*y + b*z

data = fit_guess(xx, yy, zz, 1, 1) + np.random.normal(0, 1, size=np.shape(xx)) + 0.1*np.sin(xx)*np.cos(yy)


xdata = np.vstack((xx.ravel(), yy.ravel(), zz.ravel()))
ydata = data.ravel()


def _fit_guess(M, *args):
    x, y, z = M
    arr = fit_guess(x, y, z, *args)
    return arr

p0 = (0.1, 1)

popt, pcov = curve_fit(_fit_guess, xdata, ydata, p0)

#fit = np.zeros(data.shape)
fit = fit_guess(xx, yy, zz, *popt)

print (popt, 'A', 'alpha', 'beta')
print (pcov)

#
#for i in range(len(var2)):
#    plt.plot(var1, fit[i, :])
#    plt.scatter(var1, im[i, :])
#plt.scatter(0, 0)
##    plt.xlabel('Bu')
#plt.legend()
#plt.show()