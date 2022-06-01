#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:02:29 2021

@author: jeff
"""


from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it



def run_and_analyze(Ro, Bu, Lr, Ur, analysis_type):
    
    metric_array = np.zeros([len(Ro), len(Bu), len(Lr), len(Ur)])
      
    for i, ro in enumerate(Ro):
        for j, bu in enumerate(Bu):
            for k, lr in enumerate(Lr):
                for l, ur in enumerate(Ur):
                    exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro, bu, lr, ur)
                    exp = Simulation(ro, bu, lr, ur)
                    exp.run_sim()
                    ana = exp.analysis      
                    
                    if analysis_type == 'energy':
                        metric = ana.energy_conversion()
                    elif analysis_type == 'angle':
                        metric = ana.left_beam_angle()
                    elif analysis_type == 'flux':
                        metric = ana.flux_omega_averaged()
                    elif analysis_type == 'fwhm':
                        metric = ana.fwhm_angle()
                    metric_array[i, j, k, l] = metric
    
    return metric_array


def plot_metric_2d(Ro, Bu, Lr, Ur, analysis_type, iter_over = (0, 1)):
    metric = run_and_analyze(Ro, Bu, Lr, Ur, 'angle')
    
    var_dict = {0 : (Ro, 'Ro'), 1 : (Bu, 'Bu'), 2 : (Lr, 'Lr'), 3 : (Ur, 'Ur')}
    
    var1, var1_name = var_dict[iter_over[0]]
    var2, var2_name = var_dict[iter_over[1]]
    
    var1_len = len(var1)
    var2_len = len(var2)
    
#    plt.imshow(metric)
#    plt.colorbar()
#    plt.show()
    
    for i in range(var1_len):
        plt.scatter(var2, metric[i, :, 0, 0])
    #plt.plot(Bu, im[3, :])
    plt.show()
    return None
    
    
#plot_metric_2d(Ro, Bu, Lr, Ur, 'angle', iter_over = (0, 1))


def fit_2d_data(Ro, Bu, Lr, Ur, analysis_type, iter_over = (0, 1)):
    metric = run_and_analyze(Ro, Bu, Lr, Ur, analysis_type)
    
    if iter_over[0]==0 and iter_over[1] == 1:
        im = metric[:, :, 0, 0]
        var1 = Ro
        var2 = Bu
        var11, var22 = np.meshgrid(Ro, Bu, indexing = 'ij')
        
    elif iter_over[0]==1 and iter_over[1] == 2:
        im = metric[0, :, :, 0]
        var11, var22 = np.meshgrid(Bu, Lr, indexing = 'ij')
        var1 = Bu
        var2 = Lr
        
    elif iter_over[0]== 0 and iter_over[1] == 2:
        im = metric[:, 0, :, 0]    
        var11, var22 = np.meshgrid(Ro, Lr, indexing = 'ij')
        var1 = Ro
        var2 = Lr


    xdata = np.vstack((var11.ravel(), var22.ravel()))
    ydata = im.ravel()
#
#
    def fit_guess(x, y, A, alpha, beta):
        return A*x**alpha*y**(beta)
#    
#    def fit_guess(x, y, A, alpha, beta):
#        return A*x**alpha*np.arcsin(1./y)
    
    def _fit_guess(M, *args):
        x, y = M
        arr = fit_guess(x, y, *args)
        return arr

    p0 = (0.1, 1, 1)
    
    popt, pcov = curve_fit(_fit_guess, xdata, ydata, p0)
#
    print (popt, 'A', 'alpha', 'beta')
    print (pcov**0.5)
    fit = np.zeros(im.shape)
    fit = fit_guess(var11, var22, *popt)
    print (fit_guess(0, 0, *popt), 'fit at zero')


#    Lr = np.arange(1, 5, 0.5)
#    
#    def angle(Lr):
#        theta  = np.arcsin(1./Lr/2)
#        return theta # np.rad2deg(theta)
#    
#    
#    theta = angle(Lr)
#    print (theta)
#    plt.plot(Lr, theta)
##    plt.show()

    for i in range(len(var1)):
        plt.plot(var2, fit[i, :]*180/np.pi, label = 'Ro = {}'.format( var1[i]))
        plt.scatter(var2, im[i, :]*180/np.pi)
#    plt.scatter(0, 0)
#    plt.xlabel('Bu')
    plt.legend()
    plt.show()
    
#    for i in range(len(var2)):
#        plt.plot(var1, fit[:, i])
#        plt.scatter(var1, im[:, i])
##    plt.scatter(0, 0)
##    plt.xlabel('Bu')
##    plt.legend()
#    plt.show()
    
#    plt.imshow(fit)
#    plt.show()
    return None


#
#Ro = [0.005, 0.01, 0.02 ]
#Bu = np.array([0.5, 0.9, 1., 1.1, 1.5])
#Lr = [2.0]
#Ur = [1000.]



#
Ro = [ 0.01 ]
#Ro = [0.01, 0.02, 0.03 ]
#Ro = 0.01, 0.02
Bu = np.array([ 0.9, 1., 1.1, 1.5])
#Bu = [1.]
Lr = [1., 1.5,  3., 4.]
#Lr = [3.]
Ur = [1000.]

#Bu = np.array([0.9, 1., 1.1])
#Lr = [2., 3., 4.]




fit_2d_data(Ro, Bu, Lr, Ur, 'energy', iter_over = (1, 2))

















