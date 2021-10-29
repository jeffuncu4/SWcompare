#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:31:41 2021

@author: jeff
"""
import numpy as np
import matplotlib.pyplot as plt


phase = np.pi/9

def cos_wave(A, x, t, l, omega):
    return A*np.cos(l*x - omega*t - phase)

def sin_wave(A, x, t, l, omega):
    return A*np.sin(l*x- omega*t - phase)

def closest_ind( array, value):
    index, val =  min (enumerate(array), key=lambda x: abs(x[1]-value))
    return index


phase = np.pi/9

g = 1. # m/s**2 
f = 1. # 1/s
H = 1.

Lx = 10

nx = 50
nt = 41


wavelength = Lx/5.
l = np.pi*2/wavelength
omega = np.sqrt(g*l**2*H + f**2)
T = np.pi*2/omega

print (T)

skew_sample = 1.00
Ts = T*5*skew_sample

hw = H/10
uw = hw*f/(H*l)
vw = hw*omega/(H*l)


x = np.linspace(0, Lx, nx)
t = np.linspace(0, Ts, nt)
dt = t[1]-t[0]
dx = x[1]-x[0]
xx, tt = np.meshgrid(x, t)



h = sin_wave(hw, xx, tt, l, omega) + H
u = cos_wave(uw, xx, tt, l, omega)
v = sin_wave(vw, xx, tt, l, omega)

#u*=

plt.plot(x, h[0, :])
plt.title('space')
plt.show()

plt.scatter(t, h[:, 0])
plt.show()


u_squared = u**2 + v**2
mult = g*h**2  + h*u_squared/2 
Fx = u*mult
Fy = v*mult


int_t1 = 0
int_t2 = T*1

ind_t1 = closest_ind(t, int_t1)
ind_t2 = closest_ind(t, int_t2) 
print(ind_t1, ind_t2, (len(t)))

plt.scatter(t[ind_t1: ind_t2 ], h[ind_t1: ind_t2,  0])
plt.show()


fx_average = np.trapz(Fx[ind_t1: ind_t2 + 1 , :], dx = dt,  axis = 0)/T#(t[ind_t2 ]-t[ind_t1])
fy_average = np.trapz(Fy[ind_t1: ind_t2 + 1, :], dx = dt,  axis = 0)/T #(t[ind_t2]-t[ind_t1])

#fx_average = np.trapz(Fx, dx = dt,  axis = 0)/(t[ind_t2]-t[ind_t1])
#fy_average = np.trapz(Fy, dx = dt,  axis = 0)/(t[ind_t2]-t[ind_t1])
#
#
plt.imshow(h)
plt.show()



def anlytical_flux(uw, vw, hw):
    return (g*0.5*vw*hw*H + 1./16*hw*vw*uw**2 + 3/16*hw*vw**3)

print (anlytical_flux(uw, vw, hw)*2, (fy_average))

plt.plot(fx_average)
plt.plot(fy_average)
plt.show()

#
#
#



#
#def round_dt(dt_orig, wave_period):
#    
#    N = 8
#    save_iter = 0
#    dt_diff = 1.
#    new_dt = None
#    while dt_diff >= 0 :
#        wave_fraction = wave_period/N
#        print (N, wave_fraction)
#        dt_diff = wave_fraction - dt_orig
#        new_dt = wave_fraction
#        N+= 8
#        save_iter+= 1
#        print (save_iter)
#    return new_dt, save_iter
#
#dt_orig = 0.01
#
#
#
#wp = 1.


#
#print (round_dt(dt_orig, wp))
#
















