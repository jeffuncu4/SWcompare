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


Ro = [0.03]
Bu = [1.0]
Lr = [3.]
Ur = [1000.]



def run_and_view(Ros, Bus, Lrs, Urs):
    #flux_fields = []
    fd  = []
    vort = []
    energy_conv = []
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
        print (ana.energy_conversion(), 'energy_conversion')
#        
        print (ana.wavenumber_circle())
       # print (fwhm)
#        fft = ana.wave_propogation_spectra_omega(ana.h)
#        plt.imshow(np.abs(fft))
#        plt.show()
        ana.flux_omega_averaged()
        
        h = ana.h[-1, :, :] # - ana.h[0]
        

#        plt.imshow(h, cmap = cm.jet)
##        plt.imshow(h)
#        plt.colorbar()
#        plt.show()
        

    return h, ana.H, exp.Ld, exp.Lx

hi, H, ld, Lx = run_and_view(Ro, Bu, Lr, Ur)



from matplotlib.colors import TwoSlopeNorm

#
#save_name = ['u', 'v', 'h']
#
#field_names = {'u': u, 'v': v, 'h': h}


hi_norm =  (hi-H)/H


vmin = np.min(hi)
vmax = np.max(hi)
norm = TwoSlopeNorm(vmin=vmin, vcenter=H, vmax=vmax)
#fig =plt.figure(figsize=(8,8))


vmin = np.min(hi_norm)
vmax = np.max(hi_norm)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

dim  = Lx/ld

extent = [-dim,dim, -dim,dim]

plt.imshow(hi_norm, aspect = 1, extent = extent, cmap = 'bwr', vmin  = vmin, vmax = vmax, norm = norm)

#plt.imshow(hi, aspect = 1, extent = extent, cmap = 'bwr', vmin  = vmin, vmax = vmax, norm = norm)
#plt.xlabel('x (km)')

plt.xlabel(r'$x/L_d$', fontsize = 'x-large')
plt.ylabel(r'$y/L_d$', fontsize = 'x-large')
   # plt.clim(-hw*xlima,  hw*xlima)
   # plt.clim(2*(minh-H)/H,-2*(minh-H)/H)

clb = plt.colorbar()
clb.ax.set_title(r'$\eta$/H', fontsize = 'x-large')

plt.show() 






