#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:05:29 2022

@author: jeff

THis code should turn BUrger number into a group speed. the initial rossby number into Adjust rossbyumber or max rossbymber

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



def calculate_cg(Bu, Lr):
#    exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(0.01, 0.9,1.)
    exp = Simulation(0.01, 0.9, 1., 1000.)
    exp.run_sim()
    ana = exp.analysis
    L = ana.L
    f = ana.f    
    
    cg_array = np.zeros([len(Bu), len(Lr)])
    for i, bu in enumerate(Bu):
        for j, lr in enumerate(Lr):
            cg_array[i, j] = f*L*np.pi*2*lr*bu*(1./(1 + lr**2*bu*4*np.pi**2))**0.5
    
    return cg_array







Ro = [0.01, 0.02, 0.03 ]
Bu = np.array([0.9, 1., 1.1, 1.5])
Lr = [1., 1.5, 3., 4.]
Ur = [1000.]


analysis_type = 'energy'
metric = run_and_analyze(Ro, Bu, Lr, Ur, analysis_type)
cg_array = calculate_cg(Bu, Lr)

plt.plot(cg_array.flatten())
plt.ylabel('group speed')
plt.xlabel('Bu, Lr pairs')
plt.show()

for i in range(len(Ro)):
    plt.plot(cg_array.flatten(), metric[i, :, :, 0].flatten(), label = 'Ro = {}'.format(Ro[i]))

plt.ylabel('energy conversion')
plt.xlabel('group speed')
plt.legend()
plt.show()














