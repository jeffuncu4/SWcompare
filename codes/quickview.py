#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:48:00 2021

@author: jeff
"""


from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it



Ro = [0.01 ]
Bu = np.array([0.9])
Lr = [2.0]
Ur = [1000.]


def run_and_flux(Ros, Bus, Lrs, Urs):
    #flux_fields = []
    fd  = []
    vort = []
    #vort = []
    for ro, bu, lr, ur in it.product(Ros, Bus, Lrs, Urs):
        exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
        exp = Simulation(ro, bu, lr, ur)
        exp.run_sim()
        fx, fy, flux_diff = exp.analysis.flux_omega_averaged()
        fd.append(flux_diff)
        Fm = np.sqrt(fx**2 +fy**2)
        
        plt.imshow(Fm)
        plt.show()

    return None


run_and_flux(Ro, Bu, Lr, Ur)