#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:48:23 2021

@author: jeff
"""
import pickle
import numpy as np
import h5py
import numpy.fft as fft
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
#import scipy.fftpack as sfft
#
class Analysis:
    def __init__(self, Ro, Bu, Lr, Ur):
    ###PARAMETERS
    
        #non dimensional numbers
        self.Ro = Ro
        self.Bu  = Bu
        self.Lr = Lr
        self.Ur = Ur
        

        self.bash_script_dir = '../bash_scripts/'
        ### SIMULATION LOGISITICS
        self.exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(self.Ro, self.Bu, self.Lr, self.Ur)
        self.exp_dir = '../experiments/' + self.exp_name + '/'
        self.data_file_name = self.exp_dir + 'data/data_s1.h5'
        self.vortex_dir = '../vortices/'
        self.vortex_name = 'Ro{}Bu{}_vortex.h5'.format(self.Ro, self.Bu)

        self.par_dict = pickle.load(open(self.exp_dir + 'IC/parameters.pkl', 'rb'))
        
        self.f = self.par_dict['f']
        self.L = self.par_dict['L']
        
        
        self.g = self.par_dict['g']
        self.Ly = self.par_dict['Ly']

        self.omega = self.par_dict['omega']
        self.l = self.par_dict['l']
        self.wavelength = self.par_dict['wavelength']
        self.H = self.par_dict['H']
        
        
        self.c_rot = self.omega/self.l
        
        data_file = h5py.File(self.data_file_name, mode = 'r')
        self.u = np.array(data_file['tasks']['u'])
        self.v = np.array(data_file['tasks']['v'])
        self.h = np.array(data_file['tasks']['h'])
        self.time_axis = np.array(data_file['scales']['sim_time'])
        self.x_axis = np.array(data_file['scales']['x']['1.0'])
        self.y_axis = np.array(data_file['scales']['y']['1.0'])
        self.dt = self.time_axis[1]-self.time_axis[0] # note this is not the dt the simulation ran at
        self.dx = self.x_axis[1] - self.x_axis[0]
        self.nt = len(self.time_axis)
        self.nx = len(self.x_axis)
        self.ny = len(self.y_axis)
        self.shape = np.shape(self.u)
        
        
        self.k_array = fft.fftshift(fft.fftfreq(self.nx, d=self.dx))*np.pi*2
        self.l_array = fft.fftshift(fft.fftfreq(self.ny, d=self.dx))*np.pi*2
        self.omega_array = fft.fftshift(fft.fftfreq(self.nt, d=self.dt)*2 *np.pi)
        print (np.shape(self.u))
    
    def vorticity(self):
        vx = np.gradient(self.v[0], axis = 0)/self.dx
        uy = np.gradient(self.u[0], axis = 1)/self.dx
        return vx - uy
    
    def flux_bands_plot(self):
        t_passed_vortex = self.Ly/self.c_rot*1# possible error if wave hasnt reached full domain
        t1 = self.closest_ind(self.time_axis, t_passed_vortex)
        wave_period = np.pi*2/self.omega
        wave_average = wave_period*1
        
        t2 = self.closest_ind(self.time_axis, t_passed_vortex + wave_average)
        
        
        return F_left, F_right

    
    def flux_omega_averaged(self):
        t_passed_vortex = self.Ly/self.c_rot*1# possible error if wave hasnt reached full domain
        t1 = self.closest_ind(self.time_axis, t_passed_vortex)
        wave_period = np.pi*2/self.omega
        wave_average = wave_period*1
        
        t2 = self.closest_ind(self.time_axis, t_passed_vortex + wave_average) #possible error if sim time isntl longe enougy
        
        #print (np.max(self.x_axis), self. Ly/2, self.wavelength)
#        zoom_index=  200
#        x1 =  zoom_index  #self.closest_ind(self.x_axis , -self.Lx/2 + self.wavelength*2)
#        x2 =  self.nx - zoom_index  #self.closest_ind(self.x_axis , self.Lx/2 - self.wavelength*2)
        
        
        x1 =  self.closest_ind(self.x_axis , -self.Ly/2 + self.wavelength)
        x2 = self.closest_ind(self.x_axis , self.Ly/2 - self.wavelength)
        
#        x1, x2 = 100, 400
        
        # cropped fields 
        u_crop = self.u[t1:t2, x1:x2, x1:x2]
        v_crop = self.v[t1:t2, x1:x2, x1:x2]
        h_crop = self.h[t1:t2, x1:x2, x1:x2] 
        t_crop = self.time_axis[t1:t2]
        
        
        omega_array =  fft.fftshift(fft.fftfreq(len(t_crop), d=self.dt)*2 *np.pi)
        omega_ind = self.closest_ind(omega_array, self.omega)
        
        
        u_fft_time = fft.fftshift(fft.fft(u_crop, axis = 0), axes=(0,))
        v_fft_time = fft.fftshift(fft.fft(v_crop, axis = 0), axes=(0,))
        h_fft_time = fft.fftshift(fft.fft(h_crop, axis = 0), axes=(0,))
        
        N = 41
        #dt_fine = wave_period/N
        #t_fine = np.arange(t_crop[0], t_crop[0] + wave_average + dt_fine , dt_fine)
        
        t_fine = np.linspace(t_crop[0], t_crop[0] + wave_average, N)
        
        def create_wave(sin_coeff, cos_coeff, time, omega):
            return sin_coeff*np.sin(time[:, None, None]*omega) + cos_coeff*np.cos(time[:, None, None]*omega)
        
#        sin_coeff = np.imag(u_fft_time[omega_ind])
#        cos_coeff = np.real(u_fft_time[omega_ind])
        
        u_fine = create_wave(np.imag(u_fft_time[omega_ind]),
                             np.real(u_fft_time[omega_ind]), t_fine, self.omega)
        v_fine = create_wave(np.imag(v_fft_time[omega_ind]), 
                             np.real(v_fft_time[omega_ind]), t_fine, self.omega)
        h_fine = create_wave(np.imag(h_fft_time[omega_ind]), 
                             np.real(h_fft_time[omega_ind]), t_fine, self.omega) + self.H
        
        u_squared = u_fine**2 + v_fine**2
        mult = self.g*h_fine**2  + h_fine*u_squared/2 
        Fx = u_fine*mult
        Fy = v_fine*mult
        
        fx_average = np.trapz(Fx, dx = self.dt,  axis = 0)/wave_average
        fy_average = np.trapz(Fy, dx = self.dt,  axis = 0)/wave_average
        
        Fm = np.sqrt(fx_average**2 +fy_average**2)
        plt.imshow(Fm)
        plt.show()
        
        flux_diff = (np.max(Fm) - np.min(Fm))/Fm[int((x2-x1)/2),int((x2-x1)/5) ]
#        print (2)
#        print( np.min(Fm), np.max(Fm), Fm[int((x2-x1)/2),int((x2-x1)/5)])
        return flux_diff, Fm
    



    def closest_ind(self, array, value):
        index, val =  min (enumerate(array), key=lambda x: abs(x[1]-value))
        return index

    def flux(self):
        u_squared = self.u**2 + self.v**2
        mult = self.g*self.h**2  + self.h*u_squared/2 
        Fx = self.u*mult
        Fy = self.v*mult
        return Fx, Fy


    
    def energy(self):
        E = (self.h*(self.u**2 + self.v**2) + self.g*self.h**2)/2.
        return E
    
    def filter_incoming_energy(self):

        E = self.energy()
        E_wave_snap = E[-1] - E[0]   

        x1 = self.closest_ind(self.x_axis, -self.Ly/2. + self.wavelength)
        x2 = self.closest_ind(self.x_axis, self.Ly/2. - self.wavelength)
        
        E_crop = E_wave_snap[x1:x2, x1:x2] #sqaure for uniform spectra
        
        self.k_array = fft.rfftfreq(len (E_crop), d=self.dx)*np.pi*2
        self.l_array = fft.rfftfreq(len (E_crop), d=self.dx)*np.pi*2
#        plt.imshow(E_crop)
#        plt.show()
        
        E_space = fft.fftshift(fft.rfft2(E_crop), axes = 0)
#        print ( E_space)
#        print (np.max(np.abs(E_space)))

        E_space[231, :] = 0.
#        plt.imshow(np.abs(E_space))
#        plt.show()
        Ef = fft.irfft2(fft.ifftshift(E_space, axes = 0))
        
        return Ef
        
        
    def wavenumber_circle(self):
        E = self.energy()
        E_wave_snap = E[-1] - E[0]   

#        x1 = self.closest_ind(self.x_axis, -self.Ly/2. + self.wavelength)
#        x2 = self.closest_ind(self.x_axis, self.Ly/2. - self.wavelength)

        
        E_crop = E_wave_snap[:, :] #sqaure for uniform spectra
        
        k_array = fft.fftshift(fft.fftfreq(len (E_crop), d=self.dx)*np.pi*2)
        l_array = fft.fftshift(fft.fftfreq(len (E_crop), d=self.dx)*np.pi*2)
#        plt.plot(l_array)
#        plt.show()
#        plt.imshow(E_crop)
#        plt.show()
        
        E_space = fft.fftshift(fft.fft2(E_crop), axes = (0, 1))
#        plt.plot(np.abs(E_space))
#        plt.show()
#        Ei = np.abs(E_space[231, 249])
#        E_space[231, :] = 0.
        
#        Ei = np.abs(E_space[256, 276])
        Ei = np.max(np.abs(E_space[256, :]))
#        plt.imshow(np.abs(E_space))
#        plt.show()
#        
#        k_len = len(k_array)
#        extent =  np.array([k_array[len(k_array)//2], k_array[-1], k_array[0], k_array[-1]])/(2*np.pi/self.wavelength)
#        plt.imshow(np.abs(E_space[:, self.nx//2:]), extent =extent)
##        plt.scatter(xp, yp)
#        plt.colorbar()
#        plt.show()         
####        
        E_space[256, :] = 0.
        
        f_int = interpolate.interp2d(k_array, l_array, np.abs(E_space), 
                                 kind='linear')
    
        def pol2cart(rho, phi):
                x = rho * np.cos(phi)
                y = rho * np.sin(phi)
                return (x, y)
        
#        theta_array = np.arange(np.pi/2, 3/2*np.pi, np.pi*2./2./100)
        ntheta = 301
        theta_array = np.linspace(np.pi/2, 3/2*np.pi, ntheta)
        
        xp, yp = pol2cart(self.l,  theta_array)
##        print (self.l, l_array[0], l_array[-1])
#        k_len = len(k_array)
        extent =  np.array([k_array[len(k_array)//2], k_array[-1], k_array[0], k_array[-1]])/(2*np.pi/self.wavelength)
        plt.imshow(np.abs(E_space[:, self.nx//2:])/Ei, extent =extent, cmap = cm.Blues)
        plt.xlabel(r'$\lambda/\lambda_x$', fontsize = 'x-large')
        plt.ylabel(r'$\lambda/\lambda_y$', fontsize = 'x-large')
        plt.title('Energy Spectra', fontsize = 'x-large')
#        plt.scatter(xp, yp)
        cbar = plt.colorbar()
        cbar.set_label(r'$E/E_i$', fontsize = 'x-large')
        
        
        plt.show()        
#        
                

        valp = f_int(xp, yp)
        valp = []
        
        for xi, yi in zip(xp, yp):
            valp.append(f_int(xi, yi)[0])    
            

        valp = np.array(valp)
        left_lobe = valp[0: ntheta//2 +1]
        right_lobe = valp[ntheta//2 + 1:]
        
#        plt.plot(theta_array[ntheta//2 + 1:], right_lobe)
#        plt.plot(theta_array[0: ntheta//2 +1], left_lobe)
#        plt.show()
#        
        max_left_lobe = np.max(left_lobe)
#        print (valp[ntheta//2 + 1:], 'max val p')
        max_right_lobe = np.max(right_lobe)
        
        max_ind_left = np.where(valp == max_left_lobe)[0][0]
        max_ind_right = np.where(valp == max_right_lobe)[0][0]
#        
#        fwhm_left = self.FWHM(left_lobe, theta_array)
#        fwhm_right = self.FWHM(right_lobe, theta_array)
#        print (fwhm_left )
        fwhm_left = 0
        fwhm_right = 0     
        
#        print (max_ind_left)
        max_ind = np.where(valp == np.max(valp))[0][0]
        

#        print (theta_array[max_ind], np.max(valp)/Ei,  max_ind)
#        plt.title('Scattered Energy', fontsize = 'x-large')
#        plt.plot((theta_array - np.pi)*180/np.pi, valp/Ei)
#        plt.xlabel(r'$\theta$', fontsize = 'x-large')
#        plt.ylabel(r'$E/E_i$', fontsize = 'x-large')
#        plt.plot()
#        plt.show()
##
##        
        return np.max(valp)/Ei , theta_array[max_ind_left], theta_array[max_ind_right], fwhm_left

    def group_speed(self):
        denom = (1 + self.Lr**2*self.Bu*4*np.pi**2)**0.5
        numer = self.f*self.L*np.pi*2*self.Lr*self.Bu
        return numer/denom
    
    def FWHM(self, x, t):
        half_max = np.max(x)/2

    
        max_ind = np.where(np.max(x)==x)[0][0]
    
        left = x[:max_ind]
        right = x[max_ind:]
        t_left = t[:max_ind]
        t_right = t[max_ind:]
        ind_right = self.closest_ind(right, half_max)
        ind_left = self.closest_ind(left, half_max)
    
        width = t_right[ind_right] -t_left[ind_left]
        
        return width



    def energy_conversion(self):
        ec, _, _, _ = self.wavenumber_circle()
        return ec
    def left_beam_angle(self):
        _, angle, _, _ = self.wavenumber_circle()
        return np.pi -  angle

    def right_beam_angle(self):
        _, _, angle, _ = self.wavenumber_circle()
        return angle

    def fwhm_angle(self):
        _, _, _, fwhm_left  = self.wavenumber_circle()
        return fwhm_left
    


    def wave_propogation_spectra(self, data):
        
#        wavelengths_crop = 6
#        x1 = -self.wavelength*wavelengths_crop
#        x2 = self.wavelength*wavelengths_crop
#        
#        y1 = -self.wavelength*wavelengths_crop
#        y2 = self.wavelength*wavelengths_crop
        
        x1 = 0
        x2 = self.Ly/2 - self.wavelength
        
        y1 = -1*x2/2
        y2 = 1*x2/2
        
        
        y1_vortex_index, y1_closest_val = min(enumerate(self.x_axis), 
                                  key=lambda x: abs(x[1]-y1))

        y2_vortex_index, y2_closest_val = min(enumerate(self.x_axis), 
                                  key=lambda x: abs(x[1]-y2))

        
        
        
        x1_vortex_index, x1_closest_val = min(enumerate(self.x_axis), 
                                  key=lambda x: abs(x[1]-x1))

        x2_vortex_index, x2_closest_val = min(enumerate(self.x_axis), 
                                  key=lambda x: abs(x[1]-x2))

                
        h_filtered, h_vortex = self.filter_vortex(data)
        h_filtered = h_filtered[:, y1_vortex_index:y2_vortex_index + 1, x1_vortex_index:x2_vortex_index + 1]
        h_fft_time = fft.fftshift(fft.fft(h_filtered, axis = 0), axes=(0,))
        
        
        t_passed_vortex = self.Ly/self.c_rot*0.9
        t_passed_vortex_index, t_passed_closest_val = min(enumerate(self.time_axis), 
                                  key=lambda x: abs(x[1]-t_passed_vortex))
        print (t_passed_vortex_index, 't_passed vortex index HERE')
        
        

        plt.imshow(np.real(h_filtered[-1]))
        plt.title('cropped version of  h filtered')
        plt.show()
        
        
        cropped_omega_array =  fft.fftshift(fft.fftfreq(self.nt - t_passed_vortex_index, d=self.dt)*2 *np.pi)
        
        negative_indices  = np.where(cropped_omega_array <0)
        lst_neg_ind = negative_indices[0][-1]
        
        h_fft_time[:lst_neg_ind+1 ] = 0.
        h_fft_time[lst_neg_ind+2:] *= 2
        
        h_fft_time_space = fft.fftshift(fft.fft2(h_fft_time), axes=(1,2))
        return h_fft_time_space
    
    
    
    
    
    
    
    
    
    
    
    def filter_vortex(self, data):
        
        # I need to crop to the time after the wave has crossed the vortex
        t_passed_vortex = self.Ly/self.c_rot*0.9
        t_passed_vortex_index, t_passed_closest_val = min(enumerate(self.time_axis), 
                                  key=lambda x: abs(x[1]-t_passed_vortex))
        print (t_passed_vortex_index, 't_passed vortex index HERE')
        
        
        
        cropped_omega_array =  fft.fftshift(fft.fftfreq(self.nt - t_passed_vortex_index, d=self.dt)*2 *np.pi)
        
        #T =  np.pi*2/self.omega
       # period_index_length = int(T/self.dt)
        
        data = data[t_passed_vortex_index:, :, :]
        
        h_fft_time = fft.fftshift(fft.fft(data, axis = 0), axes=(0,))
        
        omega1_ind, closest_val = min(enumerate(cropped_omega_array), 
                                  key=lambda x: abs(x[1]+self.omega/3.))
        
        omega2_ind, closest_val = min(enumerate(cropped_omega_array), 
                                  key=lambda x: abs(x[1]-self.omega/3.))
        print (omega1_ind, omega2_ind)
        h_vortex_fft_time = np.zeros(np.shape(h_fft_time), dtype = complex)
        
        h_vortex_fft_time[omega1_ind:omega2_ind, :, :]  = h_fft_time[omega1_ind:omega2_ind, :, :] 
        h_fft_time[omega1_ind:omega2_ind, :, :] = 0.0
        
        h_vortex = fft.ifft(fft.ifftshift(h_vortex_fft_time, axes=(0,)), axis = 0)
        h_filtered = fft.ifft(fft.ifftshift(h_fft_time, axes=(0,)), axis = 0)
        
        return h_filtered, h_vortex

    def wave_propogation_spectra_omega(self,  data):

        t_passed_vortex = self.Ly/self.c_rot*0.9
        t_passed_vortex_index, t_passed_closest_val = min(enumerate(self.time_axis), 
                                  key=lambda x: abs(x[1]-t_passed_vortex))
        print (t_passed_vortex_index, 't_passed vortex index HERE')
        
        
        cropped_omega_array =  fft.fftshift(fft.fftfreq(self.nt - t_passed_vortex_index, d=self.dt)*2 *np.pi)
        
        
        h_fft_time_space = self.wave_propogation_spectra( data)
        omega_ind, closest_val = min(enumerate(cropped_omega_array), 
                                     key=lambda x: abs(x[1]-self.omega))
        print ('im omega_ind', omega_ind)
        return h_fft_time_space[omega_ind]
    
    def filter_pizza_slice(self, omega_val, rho1, rho2, phi1, phi2):
        spectra_omega = self.wave_propogation_spectra(self.omega)
        
        kk, ll = np.meshgrid(self.k, self.l)
        
        print (np.min(np.arctan2(ll, kk)))
    
        def band_index(rho1, rho2, phi1, phi2):  
            index_search_i = []
            index_search_j = []
            for i in range(len(self.k)):
                for j in range(len(self.l)):
                    rad  =  (kk[i, j]**2 + ll[i, j]**2)**0.5
                   # print (rad)
                    if (rho1) < rad < (rho2):
                        #print ('check rho')
                        angle = np.arctan2(ll[i, j],kk[i, j]) + np.pi
                        if phi1 <= angle <= phi2:
                            index_search_i.append(i)
                            index_search_j.append(j)
            return np.array([index_search_i]), np.array([index_search_j])
    
        band_i, band_j = band_index(rho1, rho2, phi1, phi2)
        
#        plt.scatter(band_i, band_j)
#        plt.xlim(0, len(self.k))
#        plt.ylim(0, len(self.l))
#        plt.show()
        
        omega_ind, closest_val = min(enumerate(self.omega), 
                             key=lambda x: abs(x[1]-omega_val))
                
        h_slice_fft_time_space = np.zeros(np.shape(spectra_omega), dtype = complex)
        for i, j in zip(band_i, band_j):
            h_slice_fft_time_space[omega_ind, i, j] = spectra_omega[omega_ind, i, j]
            spectra_omega[omega_ind, i, j] = 0.
    

        

        h_slice_fft_time = fft.ifft2(fft.ifftshift(h_slice_fft_time_space, axes=(1,2)))
        h_slice = fft.ifft(fft.ifftshift(h_slice_fft_time, axes=(0,)), axis = 0)
        
        filtered_fft_time = fft.ifft2(fft.ifftshift(spectra_omega, axes=(1,2)))
        h_filtered = fft.ifft(fft.ifftshift(filtered_fft_time, axes=(0,)), axis = 0)
        
        return h_filtered, h_slice

    def flux_bands(self):
        t_passed_vortex = self.Ly/self.c_rot*0.95# possible error if wave hasnt reached full domain
        t1 = self.closest_ind(self.time_axis, t_passed_vortex)
        wave_period = np.pi*2/self.omega
        wave_average = wave_period*1
        
        t2 = self.closest_ind(self.time_axis, t_passed_vortex + wave_average) #possible error if sim time isntl longe enougy
        
        #print (np.max(self.x_axis), self. Ly/2, self.wavelength)
#        zoom_index=  200
#        x1 =  zoom_index  #self.closest_ind(self.x_axis , -self.Lx/2 + self.wavelength*2)
#        x2 =  self.nx - zoom_index  #self.closest_ind(self.x_axis , self.Lx/2 - self.wavelength*2)
        
        
        x1 =  self.closest_ind(self.x_axis , -self.Ly/2 + self.wavelength)
        x2 = self.closest_ind(self.x_axis , self.Ly/2 - self.wavelength)
        
#        x1, x2 = 100, 400
        
        # cropped fields 
        u_crop = self.u[t1:t2, x1:x2, x1:x2]
        v_crop = self.v[t1:t2, x1:x2, x1:x2]
        h_crop = self.h[t1:t2, x1:x2, x1:x2] 
        t_crop = self.time_axis[t1:t2]
        
        
        omega_array =  fft.fftshift(fft.fftfreq(len(t_crop), d=self.dt)*2 *np.pi)
        omega_ind = self.closest_ind(omega_array, self.omega)
        
        k_array = fft.fftshift(fft.fftfreq(len(u_crop[0, :, 0]), d=self.dx))*np.pi*2
        l_array = fft.fftshift(fft.fftfreq(len(u_crop[0, 0, :]), d=self.dx))*np.pi*2
        
        u_fft_time = fft.fftshift(fft.fft(u_crop, axis = 0), axes=(0,))
        v_fft_time = fft.fftshift(fft.fft(v_crop, axis = 0), axes=(0,))
        h_fft_time = fft.fftshift(fft.fft(h_crop, axis = 0), axes=(0,))
        
        N = 41
        #dt_fine = wave_period/N
        #t_fine = np.arange(t_crop[0], t_crop[0] + wave_average + dt_fine , dt_fine)
        
        t_fine = np.linspace(t_crop[0], t_crop[0] + wave_average, N)
        
        def create_wave(sin_coeff, cos_coeff, time, omega):
            return sin_coeff*np.sin(time[:, None, None]*omega) + cos_coeff*np.cos(time[:, None, None]*omega)
        
#        sin_coeff = np.imag(u_fft_time[omega_ind])
#        cos_coeff = np.real(u_fft_time[omega_ind])
        
        u_fine = create_wave(np.imag(u_fft_time[omega_ind]),
                             np.real(u_fft_time[omega_ind]), t_fine, self.omega)
        v_fine = create_wave(np.imag(v_fft_time[omega_ind]), 
                             np.real(v_fft_time[omega_ind]), t_fine, self.omega)
        h_fine = create_wave(np.imag(h_fft_time[omega_ind]), 
                             np.real(h_fft_time[omega_ind]), t_fine, self.omega) + self.H
        
        u_squared = u_fine**2 + v_fine**2
        mult = self.g*h_fine**2  + h_fine*u_squared/2 
        Fx = u_fine*mult
        Fy = v_fine*mult
        
        
        return None

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        