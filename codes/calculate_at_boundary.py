#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:47:10 2022

@author: jeff
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp2d



def path_integral(data, x, y, path_x, path_y):
    
    npts = len(path_x)
    
    f_int = interp2d(x, y, data, kind='linear')
    
    data_int = np.zeros(npts)
    
    for i in range(npts):
        data_int[i] = f_int(path_x[i], path_y[i])
        
    dx = path_x[1:] - path_x[:-1]
    dy = path_y[1:] - path_y[:-1]
    
    ds = np.sqrt(dx**2 + dy**2)

    value = integrate.trapz(y = data_int, dx = ds)
    
    return value


if __name__ == '__main__':
    
    from simulationClass import Simulation 
    import matplotlib.pyplot as plt
    import numpy.fft as fft
    import itertools as it
    from matplotlib import cm

#    N = 11
#    x = np.linspace(0, 10, N)
#    
#    data = np.ones([N, N])
#    
#    
#    
#    path_x = np.array([0, 4.5])
#    path_y = np.array([0, 0])
#    
#    val = path_integral(data, x, x, path_x, path_y)
#    
#    print (val)  
    ro = 0.03
    bu = 1.0
    lr = 3.
    ur = 1000.

    exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro,bu,lr,ur)
    exp = Simulation(ro, bu, lr, ur)
    exp.run_sim()
    ana = exp.analysis
    h = ana.h[-1]
    _, hf = ana.flux_omega_averaged()
    print (np.shape(hf))

    x = np.arange(478)
    mid = 478//2 -0.5
    rad = 50
    path_x = np.array([mid - rad, mid - rad])
    path_y = np.array([mid - rad, mid + rad])
    val = path_integral(hf, x, x, path_x, path_y)
    print (h[10, 254:257])
    print (val)
     
    
    theta = np.linspace(-90, 90, 10)*np.pi/180
    circle_x = mid + rad*np.cos(theta)
    circle_y = mid + rad*np.sin(theta)
    
#    plt.imshow(hf)
#    plt.scatter(mid, mid)
#    plt.scatter(path_x, path_y)
#    plt.scatter(circle_x, circle_y)
#    plt.colorbar()
#    plt.show()
    
#    hf_FT = fft.rfft2(hf)
#    plt.imshow(np.abs(hf_FT))
#    plt.colorbar()
#    plt.show()
#    
    
    val_after = path_integral(hf, x, x, circle_x, circle_y)
    
    print (val_after)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    