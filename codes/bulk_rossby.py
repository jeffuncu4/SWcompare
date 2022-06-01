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



def run_and_analyze(Ro, Bu, Lr, Ur):
    
    metric_array = np.zeros([len(Ro), len(Bu), len(Lr), len(Ur)])
      
    for i, ro in enumerate(Ro):
        for j, bu in enumerate(Bu):
            for k, lr in enumerate(Lr):
                for l, ur in enumerate(Ur):
                    exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro, bu, lr, ur)
                    exp = Simulation(ro, bu, lr, ur)
                    exp.run_sim()
                    ana = exp.analysis      

    
    return ana.u[0], ana.h[0]



Ro = [ 0.02]
#Bu = np.array([0.5, 0.9, 1., 1.1, 1.5])
Bu = [1.1]
Lr = [1., 2., 3., 4.]
#Lr = [2.]
Ur = [1000.]


ro = 0.03
bu = 1.0
lr = 2.
ur = 1000.

exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro, bu, lr, ur)
exp = Simulation(ro, bu, lr, ur)
exp.run_sim()
ana = exp.analysis 
u = ana.u[0]
h = ana.h[0]
lx = ana.x_axis/1000

def sin(x, l):
    return 0.1*np.sin(x*np.pi*2/l)

plt.plot(lx, h[512//2, :])
plt.plot(lx, u[512//2, :])
plt.plot(lx, sin(lx*1000, 2.5e4))

plt.scatter(0, 0.5)
plt.show()



























