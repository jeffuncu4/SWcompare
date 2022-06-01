#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:43 2022

@author: jeff
"""

import os
import h5py
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from complex_demodulation_functions import (complex_demod_fft, 
                                            complex_demod_ifft, 
                                            find_allowed_index)




harddrive_folder = '/media/jeff/Storage2'
experiment_folder = harddrive_folder + \
'/ShallowWaterData/ShallowWaterRoBuLrUr_Oct29thCopy/experiments'



data_folder = experiment_folder + '/R01Bu1Lr5Ur1000/data/state'

##
#data_folder = experiment_folder + '/R009Bu1Lr3Ur1000/data/state'



datafile = data_folder + '/state_s1.h5'
#
#print (os.listdir(harddrive_folder))
#
data = h5py.File(datafile, 'r')

#u = np.array(data['tasks']['u'])
#v = np.array(data['tasks']['v'])
h = np.array(data['tasks']['h'])
time_axis = np.array(data['scales']['sim_time'])
x_axis = np.array(data['scales']['x']['1.0'])
y_axis = np.array(data['scales']['y']['1.0'])

print (np.shape(h))
ind_start = 250
ind_end = 310

#ind_start =300
#ind_end = 400

t_trunc = time_axis[ind_start:ind_end]

x1, x2 = 1, -1

#x = 


#
#
h_trunc = h[ind_start:ind_end]
#h_trunc = h[ind_start:ind_end, x1:x2, x1:x2]
#x_axis = x_axis[x1:x2]
plt.imshow(h_trunc[-1])
plt.colorbar()
plt.show()

h_ft = fft.fftshift(fft.fft(h_trunc, axis = 0), axes = 0)
#
#plt.plot(np.abs(h_ft[:, 0, 0]))
#plt.show()
nt = len(t_trunc)
h_ft[:nt//2 +2] = 0.0
#
#plt.plot(np.abs(h_ft[:, 0, 0]))
#plt.show()


h_trunc_filtered = np.real(fft.ifft(fft.ifftshift(h_ft, axes = 0), axis = 0))

plt.imshow(h_trunc_filtered[0])
plt.colorbar()
plt.show()

plt.imshow(h_trunc_filtered[-1])
plt.colorbar()
plt.show()



h_ft_st = complex_demod_fft(h_trunc_filtered, x_axis, t_trunc)

#
#plt.imshow(np.abs(h_ft_st[0]))
#plt.colorbar()
#plt.show()

#
##h_ft_st[0] = 0.
#
find = np.where(h_ft_st == np.max(h_ft_st))[0][0]
print (find)
plt.imshow(np.abs(h_ft_st[find]))
plt.colorbar()
plt.show()



rad0 = 128- 102
r2 = rad0 + 2
r1 = rad0 - 2
phi1 = np.pi*0.0
phi2 = np.pi*0.4
#phi1 = np.pi*1.5
#phi2 = np.pi*2.

ind_i, ind_j = find_allowed_index(h_ft_st[find], r1, r2, phi1, phi2)
#
h_filter = np.zeros(np.shape(h_ft_st), dtype = np.complex)
h_filter[:, ind_i, ind_j] = h_ft_st[:, ind_i, ind_j]

#h_ft_st[:, ind_i, ind_j] = 0.0
#h_ft_st[:, 128, :] = 0.

h_ifft = complex_demod_ifft(h_filter, x_axis, t_trunc)






#
#
plt.imshow(h_ifft[-1])
plt.colorbar()
plt.show()

#
#plt.imshow(h_ifft[0] - h_trunc[0])
#plt.colorbar()
#plt.show()
#
#

















