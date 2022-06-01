#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:14:39 2022

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

ro = 0.03
bu = 1.0
lr = 4.0
ur = 1000.0

exp = Simulation(ro, bu, lr, ur)
exp.run_sim()

Lx = exp.Lx
Ly = exp.Ly
Ld = exp.Ld
f = exp.f

ana = exp.analysis


vort  = ana.vorticity()/f


vmin = np.min(vort)
vmax = np.max(vort)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


extent = np.array([-Ly/2, Ly/2, -Lx/2, Lx/2])/Ld
plt.imshow(vort, extent = extent, cmap = 'bwr', norm = norm)
plt.ylabel(r'$y/L_d$')
plt.xlabel(r'$x/L_d$')
clb = plt.colorbar()
clb.ax.set_title(r'$\zeta/f$')

plt.show()
