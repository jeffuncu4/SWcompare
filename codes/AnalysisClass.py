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


        self.par_dict = pickle.load(open(self.exp_dir + 'IC/parameters.pkl', 'rb'))
        
        self.g = self.par_dict['g']
        self.Ly = self.par_dict['Ly']

        self.omega = self.par_dict['omega']
        self.l = self.par_dict['l']
        self.wavelength = self.par_dict['wavelength']
        
        self.c_rot = self.omega/self.l
        
        data_file = h5py.File(self.data_file_name, mode = 'r')
        self.u = np.array(data_file['tasks']['u'])
        self.v = np.array(data_file['tasks']['v'])
        self.h = np.array(data_file['tasks']['h'])
        self.time_axis = np.array(data_file['scales']['sim_time'])
        self.x_axis = np.array(data_file['scales']['x']['1.0'])
        self.y_axis = np.array(data_file['scales']['y']['1.0'])
        self.dt = self.time_axis[1]-self.time_axis[0]
        self.dx = self.x_axis[1] - self.x_axis[0]
        self.nt = len(self.time_axis)
        self.nx = len(self.x_axis)
        self.ny = len(self.y_axis)
        self.shape = np.shape(self.u)
        
        
        self.k_array = fft.fftshift(fft.fftfreq(self.nx, d=self.dx))*np.pi*2
        self.l_array = fft.fftshift(fft.fftfreq(self.ny, d=self.dx))*np.pi*2
        self.omega_array = fft.fftshift(fft.fftfreq(self.nt, d=self.dt)*2 *np.pi)
        print (np.shape(self.u))
        

    def filter_vortex(self, data):
        
        # I need to crop to the time after the wave has crossed the vortex
        t_passed_vortex = self.Ly/self.c_rot*0.9
        t_passed_vortex_index, t_passed_closest_val = min(enumerate(self.time_axis), 
                                  key=lambda x: abs(x[1]-t_passed_vortex))
        print (t_passed_vortex_index, 't_passed vortex index HERE')
        
        
        
        cropped_omega_array =  fft.fftshift(fft.fftfreq(self.nt - t_passed_vortex_index, d=self.dt)*2 *np.pi)
        
        
        
        
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
#    
#    def filter_pizza_slice(self, omega_sim, omega_val, rho1, rho2, phi1, phi2):
#        spectra_omega = self.wave_propogation_spectra(omega_sim)
#        
#        kk, ll = np.meshgrid(self.k, self.l)
#        
#        print (np.min(np.arctan2(ll, kk)))
#    
#        def band_index(rho1, rho2, phi1, phi2):  
#            index_search_i = []
#            index_search_j = []
#            for i in range(len(self.k)):
#                for j in range(len(self.l)):
#                    rad  =  (kk[i, j]**2 + ll[i, j]**2)**0.5
#                   # print (rad)
#                    if (rho1) < rad < (rho2):
#                        #print ('check rho')
#                        angle = np.arctan2(ll[i, j],kk[i, j]) + np.pi
#                        if phi1 <= angle <= phi2:
#                            index_search_i.append(i)
#                            index_search_j.append(j)
#            return np.array([index_search_i]), np.array([index_search_j])
#    
#        band_i, band_j = band_index(rho1, rho2, phi1, phi2)
#        
##        plt.scatter(band_i, band_j)
##        plt.xlim(0, len(self.k))
##        plt.ylim(0, len(self.l))
##        plt.show()
#        
#        omega_ind, closest_val = min(enumerate(self.omega), 
#                             key=lambda x: abs(x[1]-omega_val))
#                
#        h_slice_fft_time_space = np.zeros(np.shape(spectra_omega), dtype = complex)
#        for i, j in zip(band_i, band_j):
#            h_slice_fft_time_space[omega_ind, i, j] = spectra_omega[omega_ind, i, j]
#            spectra_omega[omega_ind, i, j] = 0.
#    
#
#        
#
#        h_slice_fft_time = fft.ifft2(fft.ifftshift(h_slice_fft_time_space, axes=(1,2)))
#        h_slice = fft.ifft(fft.ifftshift(h_slice_fft_time, axes=(0,)), axis = 0)
#        
#        filtered_fft_time = fft.ifft2(fft.ifftshift(spectra_omega, axes=(1,2)))
#        h_filtered = fft.ifft(fft.ifftshift(filtered_fft_time, axes=(0,)), axis = 0)
#        
#        return h_filtered, h_slice
#    
#    def wavenumber_circle(self):
#        h_fft_time_space = self.wave_propogation_spectra_omega( self.h)
#
#        f_int = interpolate.interp2d(self.k, self.l, np.abs(h_fft_time_space), kind='linear')
#        
#        def pol2cart(rho, phi):
#            x = rho * np.cos(phi)
#            y = rho * np.sin(phi)
#            return (x, y)
#    
#        theta_array = np.arange(0, np.pi*2, np.pi*2./1000)
#        xp, yp = pol2cart(self.l,  theta_array)
#        valp = f_int(xp, yp)
#        valp = []
#        
#        for i in range(len(yp)):
#            valp.append(f_int(xp[i], yp[i])[0])
#        
#        return theta_array, np.array(valp)
#            

    def flux_vector(self):
        
        u_filtered_cropped, temp = self.filter_vortex(self.u)
        v_filtered_cropped, temp = self.filter_vortex(self.v)
        h_filtered_cropped, temp = self.filter_vortex(self.h)
        
        u_squared = u_filtered_cropped**2 + v_filtered_cropped**2
        mult = self.g*h_filtered_cropped**2  + h_filtered_cropped*u_squared/2 
        Fx = u_filtered_cropped*mult
        Fy = v_filtered_cropped*mult
        flux = np.array([Fx, Fy])
        return flux
    
    def flux_magnitude(self):
        flux = self.flux_vector()
        return np.linalg.norm(flux, axis= 0)
    
    def flux_wave_averaged(self, omega):
        fx, fy = self.flux_vector()
        t_index = 0
        wave_period = np.pi*2/omega
        t_length = int(wave_period/(self.dt)) # this shouold be an input
        print ('wave_period from flux wave_averaged:   ', wave_period,t_index, t_length, (self.dt) )
        fx = fx[t_index: t_index + t_length + 1 , :, :]
        fy = fy[t_index: t_index + t_length + 1]
        fx_average = np.trapz(fx, dx = self.dt,  axis = 0)
        fy_average = np.trapz(fy, dx = self.dt,  axis = 0)
        
        return np.array([fx_average, fy_average])/(t_length*self.dt)
    
    def flux_wave_averaged_mag(self,  omega):
        return np.linalg.norm(self.flux_wave_averaged( omega), axis = 0)
    
    def flux_difference(self):
        flux_mag = self.flux_wave_averaged_mag(self.omega)
        
        diff = np.max(flux_mag) - np.min(flux_mag)
        flux_incoming = 1 # palce holder
        return diff/flux_incoming
    
    def flux_angle(self):
        return None
    
#

    







