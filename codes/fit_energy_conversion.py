#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 08:40:45 2021

@author: jeff
"""


from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it


Ro = [0.005, 0.01, 0.02 ]
Bu = np.array([0.5, 0.9, 1., 1.1, 1.5])
Lr = [2.0]
Ur = [1000.]


def run_and_flux(Ros, Bus, Lrs, Urs):
    #flux_fields = []
    fd  = []
    vort = []
    energy_conv = []
    #vort = []
    for ro, bu, lr, ur in it.product(Ros, Bus, Lrs, Urs):
        exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
        exp = Simulation(ro, bu, lr, ur)
        exp.run_sim()
        
        ana = exp.analysis#        
        ec, _  = ana.wavenumber_circle()
        energy_conv.append(ec)
        max_vort = np.max(np.abs(exp.analysis.vorticity()))
        vort.append(max_vort)


    return energy_conv, vort

#


fd, vort = run_and_flux(Ro, Bu, Lr, Ur)
im = np.array(fd).reshape(3, 5)

print (im, im.shape)

#plt.imshow(im)
#plt.colorbar()
#plt.show()

#plt.plot(Bu, im[0, :])
#plt.plot(Bu, im[1, :])
#plt.plot(Bu, im[2, :])
##plt.plot(Bu, im[3, :])
#plt.show()


vort = np.array(vort).reshape(3, 5)
vortR = vort[:, 0]

print (np.shape(vortR))

bb, rr= np.meshgrid(Bu, vortR)
bb, rr= np.meshgrid(Bu, Ro)

xdata = np.vstack((bb.ravel(), rr.ravel()))
ydata = im.ravel()


def fit_guess(x, y, A, alpha, beta):
    return A*x**(alpha)*y**(beta)

def _fit_guess(M, *args):
    x, y = M
    arr = fit_guess(x, y, *args)
    return arr
    

p0 = (0.1, 1, 1)
#
popt, pcov = curve_fit(_fit_guess, xdata, ydata, p0)

print (popt, 'A', 'alpha', 'beta')
print (pcov)
fit = np.zeros(im.shape)
fit = fit_guess(bb, rr, *popt)


for i in range(3):
    plt.plot(Bu, fit[i, :], label = 'fit')
    plt.scatter(Bu, im[i, :], label  = 'flux')

plt.xlabel('Bu')
plt.legend()
plt.show()

plt.imshow(fit)
plt.show()


#
## lets predict now
#Ro2 = [0.008]
#Bu2 = np.array([0.5, 0.9, 1.0, 1.1, 1.5])
#Lr2 = [2.0]
#Ur2 = [1000.]
#
#fd2 = run_and_flux(Ro2, Bu2, Lr2, Ur2)
#fd2 = np.array (fd2)
#
#
#fit_008 = fit_guess(Bu2, Ro2, *popt)
#
#
##for i in range(3):
##    plt.plot(Ro[i]/Bu**0.5, fit[i, :], label = 'fit')
##    plt.scatter(Ro[i]/Bu**0.5, im[i, :], label  = 'flux')
##
##plt.xlabel('Bu')
##
##plt.plot(Ro2[0]/Bu2**0.5, fit_008, label = 'predicted')
##plt.scatter(Ro2[0]/Bu2**0.5, fd2, label =  'calculated')
##plt.legend()
##plt.legend()
##plt.show()
##
###
#for i in range(3):
#    plt.plot(Bu, fit[i, :], label = 'fit')
#    plt.scatter(Bu, im[i, :], label  = 'flux')
#
#plt.xlabel('Bu')
#
#plt.plot(Bu2, fit_008, label = 'predicted')
#plt.scatter(Bu2, fd2, label =  'calculated')
#plt.legend()
#plt.legend()
#plt.show()
#
#
#






