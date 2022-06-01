#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:50:54 2022

@author: jeff
"""

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
from matplotlib import cm


Ro = [0.005, 0.01, 0.02, 0.03]
Bu = [1.0]
Lr = [1.0]
Ur = [1000.]



def run(Ros, Bus, Lrs, Urs):
    #flux_fields = []
    
    energy_conv = []
    delta_t = []
    for ro, bu, lr, ur in it.product(Ros, Bus, Lrs, Urs):
        exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
        exp = Simulation(ro, bu, lr, ur)
        exp.run_sim()
#        exp.plots.view('h')
        
        
        ana = exp.analysis
        maxv = np.max(np.abs(ana.vorticity()))/ana.f
        print ('max vorticity is', maxv )
#        
#        fx, fy, flux_diff = exp.analysis.flux_omega_averaged()
#        fd.append(flux_diff)
        
#        Fm = np.sqrt(fx**2 +fy**2)
#        
#        plt.imshow(Fm)
#        plt.show()

#        E = ana.filter_incoming_energy()
        
#        plt.imshow((np.abs(E)))
#        plt.show()
#        
        Ei  = ana.energy_conversion()
        
        delta_t.append( ana.L/ana.c_rot*ana.f)
        
        energy_conv.append(Ei)

    return energy_conv, delta_t

Ei_array, delta_t = run(Ro, Bu, Lr, Ur)
print (delta_t)
Ro_bulk = [0.00981128, 0.02037097, 0.04465449, 0.07883667]

plt.plot(Ro_bulk, Ei_array)

plt.show()







