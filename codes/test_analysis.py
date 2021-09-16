#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 08:23:17 2021

@author: jeff
"""


from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt

Ro = 0.01 # must make these floats for proper naming conventions
Bu = 1
Lr = 5
Ur = 1000
#
exp1 = Simulation(Ro, Bu, Lr, Ur)
#exp1.run_sim()


a = exp1.analysis
a.crop()
h_filter, h_vortex = a.filter_vortex(a.h)

plt.imshow(np.real(h_filter[-1]))
plt.colorbar()
plt.show()