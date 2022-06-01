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
                    metric_array[i, j, k, l] = metric
    
    return metric_array

Ro = [[0.03]]

#Ro = [[0.005], [0.01], [0.02], [0.03]]
#Ro = [0.005, 0.01, 0.02, 0.03]
#Bu = [[0.9], [1.], [1.1], [1.5]]
Bu = [[1.0]]
Lr = np.array( [1., 1.5, 2., 3., 4.])
#Lr = [2.]
Ur = [1000.]





#for ro in Ro:
#    metric = run_and_analyze(ro, Bu, Lr, Ur, 'angle')
#    
#    metric = metric[0, 0, :, 0]
#    def _fit_guess(x, a, b, c):
#        return  a*np.arcsin(b/x)+ c
#    
#    p0 = (0.1, 1, 1)
#    
#    #popt, pcov = curve_fit(_fit_guess, Lr, metric, p0)
#    ##
#    #print (popt, 'A', 'alpha', 'beta')
#    #print (pcov)
#    #
#    
#    
#    #Lr = np.arange(1, 5, 0.5)
#    #
#
#    
#    #
#    #theta = angle(Lr)
#    #print (theta)
#    #plt.plot(Lr, theta)
#    ##    plt.show()
#    
#    plt.scatter(Lr, 180/np.pi*metric, label = ro)
#    #plt.plot(Lr, 180/np.pi*_fit_guess(Lr, *popt))


for ro in Ro:
    for bu in Bu:
        metric = run_and_analyze(ro, bu, Lr, Ur, 'angle')

        metric = metric[0, 0, :, 0]

        plt.scatter(Lr, 180/np.pi*metric)


#Ro = [0.03]
#Bu = [1.0]
#Lr34 = [3./4]
#Ur = [1000.]
#
#metric = run_and_analyze(Ro, Bu, Lr34, Ur, 'angle')
#
#plt.scatter(Lr34, metric[0,0,0,0]*180/np.pi, color = 'green')

tr = np.array([0.8184509919171806, 0.6169861323683361, 0.5351410331766181, 0.44700015712399865, 0.3966339422367875])/2


def angle(Lr):
    theta  = 2*np.arcsin(1./Lr/2)
    return theta # np.rad2deg(theta)

plt.plot(Lr, 180/np.pi*angle(Lr), label = r'$\theta$($\Lambda$)')
plt.plot(Lr, 180/np.pi*angle(Lr)/2, label = r'$\theta$($\Lambda$)/2')
plt.plot(Lr, tr*180/np.pi , label = 'triad_resonance')
plt.title('Scattering Angle')
plt.xlabel(r'$\Lambda$')
plt.ylabel('angle in degrees')
plt.legend()
plt.show()





