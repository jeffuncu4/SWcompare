#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:03:59 2022

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it
from matplotlib import cm
import numpy.fft as fft

from complex_demodulation_functions import complex_demod_fft, complex_demod_ifft



Ro = 0.01
Bu = 1.0
Lr = 4.0
Ur = 1000.0


exp = Simulation(Ro, Bu, Lr, Ur)
exp.run_sim()
ana = exp.analysis
h = ana.h[-1, :, :]
h0 = ana.h[0, :, :]
hw = h - h0

plt.imshow(hw)
plt.title('height')
plt.colorbar()
plt.show()

h_fft = (fft.fftshift(fft.rfft2(hw)))
#h_w = fft.irfft2(fft.ifftshift(h_fft))
##
#plt.imshow(np.real(h_w))
#plt.show()

#
plt.imshow(np.abs(h_fft))
plt.title(' 2d space fft')
plt.colorbar()
plt.show()

hf_incoming = np.zeros(np.shape(h_fft), dtype = np.complex128)
hf_incoming[256, :] = h_fft[256, :]



hf_incoming = np.zeros(np.shape(h_fft), dtype = np.complex128)
hf_incoming[256,:] = h_fft[256,:]
h_incoming = np.real(fft.irfft2(fft.ifftshift(hf_incoming)))


h_fft[256, :] = 0
plt.imshow(np.abs(h_fft))
plt.title(' 2d space fft')
plt.colorbar()
plt.show()

#plt.imshow(h_incoming)
#plt.title('h_incoming')
#plt.colorbar()
#plt.show()
#
#hf_bottom = np.zeros(np.shape(h_fft), dtype = np.complex128)
#hf_bottom[257:,:] = h_fft[257:,:]
#h_bottom = np.real(fft.irfft2(fft.ifftshift(hf_bottom)))
#
#
#
#
#plt.imshow(h_bottom)
#plt.title('h_bottom')
#plt.colorbar()
#plt.show()
#
#hf_top = np.zeros(np.shape(h_fft), dtype = np.complex128)
#hf_top[:256,:] = h_fft[:256,:]
#h_top = np.real(fft.irfft2(fft.ifftshift(hf_top)))
#
#
#
#plt.imshow(h_top)
#plt.title('h_top')
#plt.colorbar()
#plt.show()







