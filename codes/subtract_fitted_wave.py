#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:38:40 2022

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm
import numpy.fft as fft

from complex_demodulation_functions import (complex_demod_fft, 
                                            complex_demod_ifft, 
                                            find_allowed_index)



Ro = 0.03
Bu = 1.0
Lr = 4.0
Ur = 1000.0


exp = Simulation(Ro, Bu, Lr, Ur)
exp.run_sim()
ana = exp.analysis
print (np.shape(ana.h))


def fcos(t, a, omega, phase, H):
    return a*np.cos(t*omega - phase) + H

#plt.plot(ana.h[:, 20, 20])
#plt.show()

trunc = 400
h_side = ana.h[trunc:, 20, 20] 

t_side = ana.time_axis[trunc:]

print (h_side - ana.H)

#plt.plot(h_side)
#plt.show()

par, pcov = curve_fit(fcos, t_side, h_side, p0=(1e-4, ana.omega*np.pi, 0, ana.H))

print (par)
h_fit = fcos(t_side, *par)
#
plt.plot(t_side, h_fit)
plt.plot(t_side, h_side)
plt.show()


h_filter = ana.h[trunc:] - ana.h[0]
h_sub =  h_filter - h_fit[:, None, None]  + ana.H
#h_sub2 =  ana.h[-1] - h_fit[-1] - ana.h[0] + ana.H



plt.imshow(h_sub[0])
plt.title('subtract wave fit')

plt.show()










#trunc_ind = 337
#h = ana.h[trunc_ind:, :, :] - ana.h[0,:, :]
#x = ana.x_axis
#t = ana.time_axis[trunc_ind:]
#
#print (exp.hw, exp.dx)
#
#nt = len(t)
#nx = ana.nx

#
#plt.imshow(h[-1])
#plt.colorbar()
#plt.show()

#
#h_ft_st = complex_demod_fft(h, x, t)
##h_ifft = complex_demod_ifft(h_ft_st, x, t)
#
##plt.imshow(h_ifft[-1])
##plt.colorbar()
##plt.show()
##
##plt.imshow((h_ifft[-1] -h[-1])/np.max(h[0]))
##plt.title('differnce')
##plt.colorbar()
##plt.show()
#
#
##
##print (np.max(h_ifft - h))
#
#
#def find_ind_val(array, val):
#    diff = np.abs(array - val)
#    #print (diff)
#    ind  = np.where(diff == np.min(diff))[0][0]
#    return ind, array[ind]
#
#
#omega = fft.fftshift(fft.fftfreq(nt, d = (t[1] - t[0])))*np.pi*2
#k = fft.fftshift(fft.fftfreq(nx, d = (x[1] - x[0])))
#omega1 = ana.omega
#
#omega1_ind, closest_val   = find_ind_val(omega, omega1)
#
#
#print (omega1_ind, closest_val, omega1)
##
##print('max_omega', omega[37], omega[38])
##
#
#
#
##
##plt.imshow(np.abs(h_ft_st[omega1_ind]))
##plt.colorbar()
##plt.title(' omega1')
##plt.show()
##
##plt.imshow(np.abs(h_ft_st[omega1_ind -5]))
##plt.title('not omega1')
##plt.colorbar()
##plt.show()
##
##
##
##
##plt.plot( np.abs(h_ft_st[:, 256, 215]))
##plt.show()
##
#
#
#h_ft_st_omega = np.copy(h_ft_st[omega1_ind])
#
#
##plt.imshow(np.abs(h_ft_st[omega1_ind]))
##plt.colorbar()
##plt.title(' unfiltered omega1')
##plt.show()
##
############## FIltering Bubble
#
#rad0 = 256 - 215
#r2 = rad0 + 2
#r1 = rad0 - 2
##phi1 = np.pi*0
##phi2 = np.pi*1/2
#phi1 = np.pi*3/2
#phi2 = np.pi*2
#
#ind_i, ind_j = find_allowed_index(h_ft_st[omega1_ind], r1, r2, phi1, phi2)
#
#h_filtered = np.zeros(np.shape(h_ft_st), dtype = np.complex)
#h_filtered[:, ind_i, ind_j] = h_ft_st[:, ind_i, ind_j] 
#h_filtered[:, 256, :] = 0.0
##h_ft_st[:, 256, 215] = 0.0
##h_ft_st[:, 256, :] = 0.0
##h_ft_st[:, :275, :] = 0.0
#
#
##################
#
#plt.imshow(np.abs(h_filtered[omega1_ind]))
#plt.colorbar()
#plt.title(' filtered')
#plt.show()
#
#
#h_ifft_filtered = complex_demod_ifft(h_filtered, x, t)
#plt.imshow(h_ifft_filtered[-1])
#plt.colorbar()
#plt.show()
