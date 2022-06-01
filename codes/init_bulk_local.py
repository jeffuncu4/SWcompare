#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 08:16:40 2021

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it


def run_and_analyze(Ro, Bu, Lr, Ur):
    
    metric_array = np.zeros([len(Ro), len(Bu), len(Lr), len(Ur)])
     
    u_max = []
    vort_max = []
    for i, ro in enumerate(Ro):
        for j, bu in enumerate(Bu):
            for k, lr in enumerate(Lr):
                for l, ur in enumerate(Ur):
                    exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro, bu, lr, ur)
                    exp = Simulation(ro, bu, lr, ur)
                    exp.run_sim()
                    ana = exp.analysis      
                    u_max.append(np.max(ana.u[0]))
                    vort_max.append(np.max(np.abs(ana.vorticity())))
                    
#                    
#                    plt.plot(ana.x_axis/1000, ana.u[0,512//2, :]/exp.L/exp.f)
#                    plt.title(exp_name)
#                    plt.ylabel('u/(Lf)')
#                    plt.xlabel('x (km)')
#                    plt.show()
    
    return np.array(u_max), np.array(vort_max)


Ro = [0.005, 0.01, 0.02, 0.03]
Bu = [1.1]
Lr = [2.]
Ur = [1000.]

u_max, vort_max = run_and_analyze(Ro, Bu, Lr, Ur)
exp = Simulation(0.01, 1.1, 2., 1000.)


plt.plot(Ro, u_max/exp.L/exp.f, label = 'bulk')
plt.plot(Ro, vort_max/exp.f, label = 'local')
plt.plot(Ro, Ro, label = 'init')
plt.legend()
plt.xlabel('init Ro')
plt.show()

print (vort_max/exp.f)
print (u_max/exp.L/exp.f, 'bulk') 