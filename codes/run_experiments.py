#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:58:02 2021

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt


def round_arange(start, stop, step):
    step_decimal_points = len(str(step).split('.')[-1])
    start_decimal_points = len(str(start).split('.')[-1])
    stop_decimal_points = len(str(stop).split('.')[-1])
    max_dec_points = max([start_decimal_points, stop_decimal_points, step_decimal_points])

    num_list =  np.arange(start, stop, step)
    num_rounded_list = []
    for num in num_list:
        rounded_num = round(num, max_dec_points)
        num_rounded_list.append(float(str(rounded_num).rstrip('0')))
    return np.array(num_rounded_list)


#
Ro = 0.01 # must make these floats for proper naming conventions
Bu = 1.
Lr = 4.
Ur = 1000.

exp = Simulation(Ro, Bu, Lr, Ur)
exp.run_sim()




ana = exp.analysis

flux = ana.flux_omega_averaged()
#
#plt.imshow(flux[1])
#plt.colorbar()
#plt.show()
#plt.imshow(flux[0])
#plt.colorbar()
#plt.show()

plt.imshow(np.sqrt(flux[1]**2 + flux[0]**2))
plt.colorbar()
plt.show()
#
#plt.imshow((flux[0]**2 + flux[1]**2)**0.5)
#plt.show()

#for ro in  (0.01,  0.03, 0.05):
#    for lr in (1, 2, 3, 4, 5, 0.5, 0.33):
#        exp = Simulation(ro, Bu, lr, Ur)
#        exp.run_sim()
        
#
#for lr in np.arange(1, 6, 1):
#    exp = Simulation(Ro, Bu, lr, Ur)
#    exp.run_sim()    


#exp1 = Simulation(Ro, Bu, Lr, Ur)

#exp1.run_sim()



#exp1.run_vortex_sim(run_if_created=True)
#exp1.run_main_sim(run_if_created=True)
#exp1.make_uvh_videos(run_if_created=True)
#exp1.create_sim(run_if_created=True)
#exp1.run_vortex_sim(run_if_created=True)

#exp1.run_vortex_sim(run_if_created=True)


##
exp.plots.view('h')
#exp1.plots.view('v')
#exp1.plots.view('u')
#print (exp1.analysis.u)
#
#print (type(exp1.analysis.u))

#
#
#theta_array, valp = exp1.analysis.wavenumber_circle()
#
#
#plt.plot(theta_array, valp)
#plt.show()

#
#df, dv = exp1.analysis.filter_vortex(exp1.analysis.h)
#
#plt.imshow(np.real(df[-1]))
#plt.title('vortex filter')
#plt.colorbar()
##plt.clim(-5e-6, 5e-6)
#plt.show()
#
#ana = exp1.analysis
#print (exp1.omega)
##
#plt.imshow(ana.flux_wave_averaged_mag(exp1.omega)[:, :450])
#
#plt.show()
#
#spec = ana.wave_propogation_spectra_omega(ana.h)
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






        