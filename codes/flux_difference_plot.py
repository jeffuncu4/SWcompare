#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:02:41 2021

@author: jeff
"""
from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt




#Ro = 0.01 # must make these floats for proper naming conventions
#Bu = 1.
#Lr = 4.
#Ur = 1000.
#
#exp = Simulation(Ro, Bu, Lr, Ur)
#exp.run_sim()
#
#
#ana = exp.analysis
#
#flux = ana.flux_omega_averaged()
#
#plt.imshow(flux)
#plt.colorbar()
#plt.show()


Bu =1.
Ur = 1000.
lrs = (2., 0.5, )
ros = (0.005, 0.01)

#flux_difs = np.zeros[len(ros), len(lrs)]

for i, ro in  enumerate(ros):
    for j, lr in enumerate(lrs):
        exp = Simulation(ro, Bu, lr, Ur)
        exp.run_sim()
        fx, fy = exp.analysis.flux_omega_averaged()
        F = (fx**2 +fy**2)**0.5
        plt.imshow(F)
        plt.colorbar()
        plt.show()
        
#
#for i, ro in enumerate(ros):
#    plt.plot(lrs, flux_difs[i, :], label = str(ro))
#    
#plt.legend()
#plt.title('flux magnification')
#plt.xlabel('L/wavelength')
#plt.ylable(' max (flux) - min(flux) over incoming flux')
#plt.show()