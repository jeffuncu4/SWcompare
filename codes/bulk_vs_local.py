#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 08:01:21 2021

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
                    v0 = ana.vorticity()

    
    return ana.u[0], ana.h[0], v0



Ro = [ 0.02]
#Bu = np.array([0.5, 0.9, 1., 1.1, 1.5])
Bu = [1.1]
Lr = [1., 2., 3., 4.]
#Lr = [2.]
Ur = [1000.]


ro = 0.03
bu = 1.0
lr = 1.
ur = 1000.

exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro, bu, lr, ur)
exp = Simulation(ro, bu, lr, ur)
exp.run_sim()
ana = exp.analysis 
v0 = ana.vorticity()
u = ana.u[0]
h = ana.h[0]-ana.H
lx = ana.x_axis/1000

vmax  = np.max(np.abs(v0))
umax  = np.max(np.abs(u))
hmax  = np.max(np.abs(h))

def sin(x, l):
    return np.sin(x*np.pi*2/l)

plt.plot(lx, h[512//2, :]/hmax, label = 'height')
plt.plot(lx, u[512//2, :]/umax, label = 'velocity')
plt.plot(lx, sin(lx*1000, 2.5e4), label = 'sine curve')
plt.plot(lx, v0[512//2, :]/vmax, label  = 'vorticity')


plt.scatter(0, 0.5)
plt.xlabel('x (km)')
plt.legend()
plt.show()





























