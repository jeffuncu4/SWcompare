#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:58:02 2021

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt

Ro = 0.1 # must make these floats for proper naming conventions
Bu = 1.
Lr = 1.
Ur = 1000.


#for ro in np.arange(1000, 20000, 11.6):
#    exp = Simulation(ro, Bu, Lr, Ur)

exp1 = Simulation(Ro, Bu, Lr, Ur)

exp1.run_sim()
#exp1.create_sim(run_if_created=True)
#exp1.run_vortex_sim(run_if_created=True)

#exp1.run_vortex_sim(run_if_created=True)


##
#exp1.plots.view('h')
#exp1.plots.view('v')
#exp1.plots.view('u')
#print (exp1.analysis.u)
#
#print (type(exp1.analysis.u))
#
df, dv = exp1.analysis.filter_vortex(exp1.analysis.h)

plt.imshow(np.real(df[-1]))
plt.show()

ana = exp1.analysis
print (exp1.omega)

#plt.imshow(ana.flux_wave_averaged_mag(300, exp1.omega)[:, 335:512])
#plt.show()

#spec = ana1.wave_propogation_spectra_omega(ana1.h)
#
#plt.imshow(np.abs(spec))
#plt.show()

#################
#
#Lr = 1./np.arange(1, 6)
#Ro = np.arange(0.01, 0.1, 0.01)
#
#Bu = 1
#Ur = 1000
#
#flux_difference = np.array([len(Lr), len(Ro)])
#mean_angle = np.array([len(Lr), len(Ro)])
#
#
#
#for i, lr in enumerate(Lr):
#    for j, ro in enumerate(Ro):
#        experiment = Simulation(Ro, Bu, Lr, Ur).run_sim()
#        exp_analysis = experiment.analysis
#        flux = exp_analysis.flux()
#        max_flux, min_flux  = np.max(flux) , np.min(flux)
#        flux_difference[i, j] = max_flux -  min_flux
#        
#        wavenumber_circle = exp_analysis.wavenumber_circle()
#        mean_angle[i, j] = np.mean(wavenumber_circle)
#        
#        
## do the same thing for finding peaks, standard deviation?
#        
#        
#
#
#
#
#






        