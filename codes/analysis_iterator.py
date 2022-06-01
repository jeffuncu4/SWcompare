#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:55:53 2021

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt



def analysis_iterator(Ros, Bus, Lrs, Urs):
    flux_fields = []
    fd  = []
    vort = []
    for ro in Ros:
        for bu in Bus:
            for lr in Lrs:
                for ur in Urs:
                    exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
                    exp = Simulation(ro, bu, lr, ur)
                    exp.run_sim()
                    #exp.plots.view('h')

#                    plt.imshow(Fm)
#                    plt.title(exp_name + 'Flux')
#                    plt.colorbar()
#                    plt.show()
###                    
#                    fx, fy  = exp.analysis.flux_omega_averaged_2()
#                   # fx, fy = exp.analysis.flux_mag_averaged()
#                    plt.imshow(np.sqrt(fx**2 +fy**2))
#                    plt.title(exp_name + 'Flux')
#                    plt.colorbar()
#                    plt.show()

    return None


#Ro = [ 0.004 ]
#Bu = np.array([0.5, 0.9, 1., 1.1, 1.5])
#Lr = [ 1., 2., 3., 4.]
#Ur = [1000.]

Ro = [ 0.004 ]
Bu = [1.]
Lr = [2.]
Ur = [1000.]

Ro = [ 0.02 ]
Bu = [1.]
Lr = [2.]
Ur = [1000.]


analysis_iterator(Ro, Bu, Lr, Ur)


#
#Ro = [0.005, 0.01, 0.02, 0.03]
#Bu = [1.]
#Lr = [2.]
#Ur = [1000.]
#

#
#Ro = [ 0.02]
#Bu = [1.]
#Lr = [2., 1., 4.]
#Ur = [1000.]

#Ro = [0.005, 0.01, 0.02, 0.03]
#Bu = [1.]
#Lr = [2.0]
#Ur = [1000.]

#
#Ro = [ 0.02]
#Bu = [1.]
#Lr = [0.5, 1, 2, 3, 4]
#Ur = [1000.]






#
#print (im[:-1]-im[1:])
#print (im[:,:-1]-im[:, 1:])
#
#plt.imshow(im)
#plt.colorbar()
#plt.show()
#
#plt.plot(Bu, im[0, :])
#plt.plot(Bu, im[1, :])
#plt.plot(Bu, im[2, :])
#plt.show()
#
#
#
#from scipy.optimize import curve_fit
#
#rr, bb= np.meshgrid(Ro, Bu)
#print (rr)
#
#xdata = np.vstack((rr.ravel(), bb.ravel()))
#ydata = im.ravel()
#
#
#def fit_guess(x, y, A, alpha, beta):
#    return A*x**(alpha)*y**(beta)
#
#def _fit_guess(M, *args):
#    x, y = M
#    arr = fit_guess(x, y, *args)
#    return arr
#    
#
#p0 = (0.1, 1, 1)
#
#popt, pcov = curve_fit(_fit_guess, xdata, ydata, p0)
#
#print (popt)
#fit = np.zeros(im.shape)
#fit = fit_guess(rr, bb, *popt)
#
#plt.plot(Bu, fit[:, 0])
#plt.plot(Bu, im[0, :])
#plt.show()
#
#plt.imshow(fit)
#plt.show()
#
#print (fit-im.transpose())

#for ro, bu in product(Ro, Bu):
#    plt.scatter(ro, bu)

#print (vort)
#
#
#Ro = [0.005]
#Bu1 = [0.5, 0.9, 1. , 1.1, 1.5]
#Lr = [2.0]
#Ur = [1000.]
#
#
#f2, d2, vort2 = analysis_iterator(Ro, Bu1, Lr, Ur)
#

#vort.insert(0, 0)
#Ro1.insert(0, 0)
#d.insert(0, 0)
#d2.insert(0, 0)
##
#plt.scatter(Bu, np.log(d), label = 'Ro = {}'.format(vort[0]))
#plt.scatter(Bu1, np.log(d2), label = 'Ro = {}'.format(vort2[0]))
#plt.title('Lr = 2, Ur = 1000')
#plt.xlabel('Bu')
#plt.ylabel('Flux Amplification')
#plt.legend()
#plt.show()


#f2, d2 = analysis_iterator(Ro, Bu2, Lr, Ur)



#plt.scatter(Bu, d, label = 'Ro = 0.02')
#plt.scatter(Bu2, d2, label = 'Ro = 0.005')
#plt.title('Lr = 2, Ur = 1000')
#plt.xlabel('Bu')
#plt.ylabel('Flux Amplification')
#plt.legend()
#plt.show()

#plt.subplot(1,3,1)
#
#plt.imshow(f[0])
#plt.title( 'Flux')
#plt.colorbar()
#
#
#plt.subplot(1,3,2)
#
#plt.imshow(f[1])
#plt.title( 'Flux')
#plt.colorbar()
#
#
#plt.subplot(1,3,3)
#
#plt.imshow(f[2])
#plt.title( 'Flux')
#plt.colorbar()
#
#plt.show()




















