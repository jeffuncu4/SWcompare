#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:18:39 2022

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
Lr = 3.0
Ur = 1000.0


exp = Simulation(Ro, Bu, Lr, Ur)
exp.run_sim()
ana = exp.analysis
print (np.shape(ana.h))
#trunc_ind = 337
trunc_ind = 228
h = ana.h[trunc_ind:, :, :] - ana.h[0,:, :]
x = ana.x_axis
t = ana.time_axis[trunc_ind:]

print (exp.hw, exp.dx)

nt = len(t)
nx = ana.nx

#
plt.imshow(h[-100])
plt.colorbar()
plt.show()

h_fft = fft.fftshift(fft.rfft2(h[-1]), axes = (0,1))

plt.imshow(np.abs(h_fft))
plt.colorbar()
plt.show()

h_fft[256, :]  =0.


h_filt = fft.irfft2(fft.ifftshift(h_fft, axes = (0,1)))

Lx = exp.Lx
ld = exp.Ld
dim  = Lx/ld

extent = [-dim,dim, -dim,dim]

normalize = np.max(ana.h[-1, 20, 20:100]) - ana.H
print (normalize)
plt.imshow(h_filt/normalize, cmap = 'bwr', extent = extent)
plt.xlabel(r'$x/L_d$', fontsize = 'x-large')
plt.ylabel(r'$y/L_d$', fontsize = 'x-large')
clb = plt.colorbar()
clb.ax.set_title(r'$\eta/h_w$', fontsize = 'x-large')
plt.title('Scattered Wave Height', fontsize = 'x-large')
plt.show()
#


#
#
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
#h_remainder = np.zeros(np.shape(h_ft_st), dtype = np.complex)
#h_up = np.zeros(np.shape(h_ft_st), dtype = np.complex)
#h_down = np.zeros(np.shape(h_ft_st), dtype = np.complex)
#
#h_remainder = np.copy(h_ft_st)
#
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
#
#
#
#h_up[:, ind_i, ind_j] = h_remainder[:, ind_i, ind_j] 
#h_remainder[:, ind_i, ind_j] = 0.
#
##plt.imshow(np.abs(h_up[omega1_ind]))
##plt.title('h_up')
##plt.show()
#
#h_remainder[:, 256, :] = h_ft_st[:, 256, :]
#h_up[:, 256, :] = 0.0
#
#plt.imshow(np.abs(h_up[omega1_ind]))
#plt.title('h_up')
#plt.show()
#
#
#phi1 = np.pi*0
#phi2 = np.pi*1/2
##phi1 = np.pi*3/2
##phi2 = np.pi*2
#
#ind_i, ind_j = find_allowed_index(h_ft_st[omega1_ind], r1, r2, phi1, phi2)
#
#
#h_down[:, ind_i, ind_j] = h_remainder[:, ind_i, ind_j] 
#h_remainder[:, ind_i, ind_j] = 0.
#
#
#
##h_ft_st[:, 256, 215] = 0.0
##h_ft_st[:, 256, :] = 0.0
##h_ft_st[:, :275, :] = 0.0
#
#h_remainder_space = complex_demod_ifft(h_remainder, x, t)
#h_up_space = complex_demod_ifft(h_up, x, t)
#h_down_space = complex_demod_ifft(h_down, x, t)
#
##################
##
##plt.imshow(np.abs(h_filtered[omega1_ind]))
##plt.colorbar()
##plt.title(' filtered')
##plt.show()
##
##
##h_ifft_filtered = complex_demod_ifft(h_filtered, x, t)
##plt.imshow(h_ifft_filtered[-1])
##plt.colorbar()
##plt.show()
#
#
#plt.imshow(h_remainder_space[-1])
#plt.colorbar()
#plt.show()
#
#Lx = exp.Lx
#ld = exp.Ld
#dim  = Lx/ld
#
#extent = [-dim,dim, -dim,dim]
#
#plt.imshow(ana.h[-1] - ana.H)
#plt.title('h - H')
#plt.colorbar()
#plt.show()
#
#
#scattered_wave = h_up_space[-1] + h_down_space[-1]
#normalize = np.max(ana.h[-1, 20, 20:100]) - ana.H
#print (normalize)
#plt.imshow(scattered_wave/normalize, cmap = 'bwr', extent = extent)
#plt.xlabel(r'$x/L_d$', fontsize = 'x-large')
#plt.ylabel(r'$y/L_d$', fontsize = 'x-large')
#clb = plt.colorbar()
#clb.ax.set_title(r'$\eta/h_w$', fontsize = 'x-large')
#plt.title('Scattered Wave Height', fontsize = 'x-large')
#plt.show()
#
##plt.imshow(h_down_space[-1])
##plt.colorbar()
##plt.show()
##
