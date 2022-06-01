#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:21:54 2021

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools as it



def run_and_analyze(Ro, Bu, Lr, Ur, analysis_type):
    
    metric_array = np.zeros([len(Ro), len(Bu), len(Lr), len(Ur)])
      
    for i, ro in enumerate(Ro):
        for j, bu in enumerate(Bu):
            for k, lr in enumerate(Lr):
                for l, ur in enumerate(Ur):
                    exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(ro, bu, lr, ur)
                    exp = Simulation(ro, bu, lr, ur)
                    exp.run_sim()
                    ana = exp.analysis      
                    
                    if analysis_type == 'energy':
                        metric = ana.energy_conversion()
                    elif analysis_type == 'angle':
                        metric = ana.left_beam_angle()
                    elif analysis_type == 'flux':
                        metric = ana.flux_omega_averaged()
                    metric_array[i, j, k, l] = metric
    
    return metric_array

Ro_bulk = [0.00981128, 0.02037097, 0.04465449, 0.07883667]
Ro_max = [0.10335902, 0.22129639, 0.53684526, 1.27147153]

def fit_3d_data(Ro, Bu, Lr, Ur, analysis_type):
    metric = run_and_analyze(Ro, Bu, Lr, Ur, analysis_type)
    

    im = metric[:, :, :, 0]
    
#    var11, var22 = np.meshgrid(Lr, Ro, Bu)

    rr, bb, ll = np.meshgrid (Ro_bulk, Bu, Lr, indexing = 'ij')
    print (rr)

    xdata = np.vstack((rr.ravel(), bb.ravel(), ll.ravel() ))
    ydata = im.ravel()
#
#
    def fit_guess(x, y, z, A, alpha, beta, gamma):
        return A*x**(alpha)*y**(beta)*z**(gamma)
    
    def _fit_guess(M, *args):
        x, y, z = M
        arr = fit_guess(x, y, z, *args)
        return arr

    p0 = (0.1, 0.1, 0.1, -1)
    
    popt, pcov = curve_fit(_fit_guess, xdata, ydata, p0)
#
    print (popt, 'A', 'alpha', 'beta', 'gamma')
    print (pcov**0.5)

    fit = fit_guess(rr, bb, ll, *popt)
#    fit = 0
#    for i in range(len(Bu)):
#        plt.plot(Bu, fit[:, i, 1], label = Ro[i])
#        plt.scatter(Bu, metric[:, i, 1], label = Ro[i])
#    plt.legend()
#    plt.show() 
    
    return metric, fit



#
Ro = [.005, 0.01, 0.02, 0.03 ]
#Ro = [.005, 0.01, 0.02, 0.03 ]
Bu = np.array([0.9, 1., 1.1, 1.5])
#Lr = [1., 1.5, 2., 3., 4.]
Lr = [1., 1.5,  3., 4.]
#Lr = [ 1.5,  3., 4.]
Ur = [1000.]


metric, fit  = fit_3d_data(Ro, Bu, Lr, Ur, 'energy')

#
#for i, r in enumerate(Ro):
#    for j, b in enumerate(Bu):
#        for k, l in enumerate(Lr):
#            plt.scatter(r/b**0.5*l, metric[i, j, k], color = 'blue')
#            plt.scatter(r/b**0.5*l, fit[i, j, k], color = 'red')
##            plt.plot(r/b**0.5*l)

def residuals(x, y):
    fit_par = np.polyfit(x, y, deg = 1)
    fit_func = np.poly1d(fit_par)
    fit_line = fit_func(x)
    resid = np.sum((y-fit_line)**2)
    return resid, fit_line

exact_x = []
rounded_x = []
cp_x = []
cg_x = []

for i, r in enumerate(Ro_bulk):
    for j, b in enumerate(Bu):
        for k, l in enumerate(Lr):
#            exact_ijk = r**0.951/b**0.529*l**0.824
#            exact_x.append(exact_ijk)
            
            exact_ijk = r**0.955/b**0.548*l**0.916
            exact_x.append(exact_ijk)
            
            round_ijk = r/b**0.5*l
            rounded_x.append(round_ijk)
            
            cp_ijk = r**1.*(b*(1 + (l**2*b*4*np.pi**2)**-1))**-.5*l
            cp_x.append(cp_ijk)
            
            cg_d  = b*l*(1 + (l**2*b*4*np.pi**2))**-0.5*np.pi*2
            
            cg_ijk = r**1.*(cg_d)**-1*l
            cg_x.append(cg_ijk)
            
            
            
#            plt.scatter(exact_ijk, metric[i, j, k], color = 'blue', label = 'exact fit')
##            plt.scatter(round_ijk, metric[i, j, k], color = 'red', label = 'rounded fit')
##            plt.scatter(cp_ijk, metric[i, j, k], color = 'green', label = 'rotating cp fit')
#            plt.scatter(cg_ijk, metric[i, j, k], color = 'purple', label = 'rotating cg fit')
#            
#            
            plt.scatter(np.log(exact_ijk), np.log(metric[i, j, k]), color = 'blue', marker = '+', label = 'Exact fit')
            plt.scatter(np.log(round_ijk), np.log(metric[i, j, k])+1, color = 'red', marker = 'x', label = 'Rounded fit')
#            plt.scatter(cp_ijk, metric[i, j, k], color = 'green', label = 'rotating cp fit')
#            plt.scatter(np.log(cg_ijk), np.log(metric[i, j, k]), color = 'purple', label = 'rotating cg fit')
            if i==0 and j==0 and k==0:
                plt.legend(fontsize =  'x-large')
            #plt.scatter(r**1.1956/b**0.548*l**0.9166, fit[i, j, k], color = 'red')
#            plt.plot(r/b**0.5*l)

#plt.plot(exact_x, metric.flatten())
#plt.plot(rounded_x, metric.flatten())
#plt.plot(cp_x, metric.flatten())
        
resid_exact, fite = residuals(np.log(exact_x), np.log(metric.flatten()))
resid_rounded, fitr = residuals(np.log(rounded_x), np.log(metric.flatten()))
#resid_cp, fitc = residuals(cp_x, metric.flatten())      
#resid_cg, fitcg = residuals(cg_x, metric.flatten())      
#print (resid_exact, resid_rounded, resid_cp, resid_cg)
#plt.scatter(np.log(exact_x), np.log(fite))
#plt.scatter(np.log(rounded_x), np.log(fitr) +1 )

#print (resid_exact, resid_rounded, resid_cp, resid_cg)
#plt.scatter(np.log(exact_x), fite, label = 'fit_line')
#plt.scatter(np.log(rounded_x), fitr +1, label = 'fit_line')

exact_x.sort()
rounded_x.sort()
fite.sort()
fitr.sort()

plt.plot(np.log(exact_x), fite, label = 'Linear fit')
plt.plot(np.log(rounded_x), fitr +1, label = 'Linear fit')


#plt.scatter(cg_x, fitc)

plt.title('Energy conversion', fontsize = 'x-large')
plt.xlabel(r'$ln(Ro^{\alpha} Bu^{\beta} \Lambda^{\gamma})$', fontsize = 'x-large')   
plt.ylabel('$ln(E_t)$', fontsize = 'x-large')          
plt.show()

#for i, r in enumerate(Ro):
#    for j, b in enumerate(Bu):
#        for k, l in enumerate(Lr):
#            plt.scatter(1/l, metric[i, j, k], color = 'blue')
#            plt.scatter(r/b**0.5*l, fit[i, j, k], color = 'red')
#            plt.plot(r/b**0.5*l)
            
plt.show()





    
    






