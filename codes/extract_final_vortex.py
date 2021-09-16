#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:05:07 2021

@author: jeff
"""


import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys


exp_dir = '../experiments/' + sys.argv[1] + '/'

FILE = exp_dir + "IC/IC_s1.h5"



file = h5py.File(FILE, mode = 'r')
h = file['tasks']['h']
u = file['tasks']['u']
v = file['tasks']['v']
ta = file['scales']['sim_time']

#print (np.shape(h))

#plt.imshow(h[-1])
#plt.show()

#print (np.array(ta))
#
#print (list (file['tasks'].keys()))
#print (list (file['scales'].keys()))
#




saveData=True
if saveData:   
    hf = h5py.File(exp_dir + 'IC/settled_vortex.h5', 'w')
    hf.create_dataset('geoH', data=h[-1])
    hf.create_dataset('geoU', data=u[-1])
    hf.create_dataset('geoV', data=v[-1])
    hf.close()