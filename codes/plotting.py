#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:24:03 2021

@author: jeff
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:54:53 2020

@author: jeff
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:25:07 2019

@author: juncu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:59:12 2019

@author: juncu
"""

#test plotting

from dedalus import public as de
import h5py
import matplotlib.pyplot as plt
import numpy as np

import pickle
import sys



exp_dir = '../experiments/' + sys.argv[1] + '/'

par_dict = pickle.load(open(exp_dir + 'IC/parameters.pkl', 'rb'))


nx = par_dict['nx']
ny = par_dict['ny']
Lx = par_dict['Lx']
Ly = par_dict['Ly']
f = par_dict['f']
g = par_dict['g']
mu = par_dict['mu']
l = par_dict['l']
omega = par_dict['omega']
H = par_dict['H']
tau_s = par_dict['tau_s']
tau_w = par_dict['tau_w']
num_iter = par_dict['num_iter']
dt = par_dict['dt']
save_iter = par_dict['save_iter']
max_writes = par_dict['max_writes']
Ld = par_dict['Ld']



FILE = exp_dir + 'data/data_s1.h5'



file = h5py.File(FILE, mode = 'r')
h = file['tasks']['h']
u = file['tasks']['u']
v = file['tasks']['v']
ta = file['scales']['sim_time']

xbasis = de.Fourier('x', nx, interval=(-Lx/2,Lx/2), dealias=1) #dealias = 3/2 # this oworks with fourier(wihtout BC) and Chebyshev
ybasis = de.Fourier('y', ny, interval=(-Ly/2,Ly/2), dealias=1)

domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64)
x = domain.grid(0)[0]
y = domain.grid(1)[0]
#slices = domain.dist.grid_layout.slices(scales=(1,1))

t = np.arange(0, num_iter*dt, save_iter*dt)


ht = h[-1,:,:] #single timestep of h_field
htr = np.rot90(ht)

h0  = h[0,:,:] 
n = len(h)
print(n)

def pic_num(n):
    if 0<=n<=9:
        return '000%s'%(n) 
    elif 10<=n<=99:
        return '00%s'%(n)
    elif 100<=n<=999:
        return '0%s'%(n)
    elif 1000<=n<=9999:
        return str(n)


# use fig, ax tomorrow to do this properly. also automize dimensions
extent = np.array([-Ly/2, Ly/2, -Lx/2, Lx/2])/Ld


from matplotlib.colors import TwoSlopeNorm


save_name = ['u', 'v', 'h']

field_names = {'u': u, 'v': v, 'h': h}

for name in field_names:
    for i in range(0, n, 1):    
        filename = exp_dir + 'plots/' + name + '_%s'%pic_num(i)
        hi = field_names[name][i] #change this
    
    
        #vmin = np.min(hi)
        #vmax = np.max(hi)
       # norm = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
        #fig =plt.figure(figsize=(8,8))
        text = int(ta[i]/np.pi/2*f*100)/100
    
        plt.imshow(hi, aspect = 1, extent = extent, cmap = 'bwr')#, vmin  = vmin, vmax = vmax, norm = norm)
        plt.xlabel('x (km)')
        plt.title(name)
        plt.xlabel('x/Ld')
        plt.ylabel('y/Ld')
       # plt.clim(-hw*xlima,  hw*xlima)
       # plt.clim(2*(minh-H)/H,-2*(minh-H)/H)
        plt.text(1.5, 1.5, "t = %s T"%text)
        clb = plt.colorbar()
    #    clb.ax.set_title(r'$\eta$/H')
    
        #plt.show() 
        plt.savefig(filename)
        plt.close()
        
        print (i)








