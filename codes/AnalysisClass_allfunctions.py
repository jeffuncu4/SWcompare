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
import scipy.fftpack as sfft
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
    def filter_pizza_slice(self, omega_sim, omega_val, rho1, rho2, phi1, phi2):
        spectra_omega = self.wave_propogation_spectra(omega_sim)
        
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
#    
#    def wavenumber_circle(self):
#        h_fft_time_space = self.wave_propogation_spectra_omega(self.h)
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

#    def flux_vector(self):
#        
#        u_filtered_cropped, temp = self.filter_vortex(self.u)
#        v_filtered_cropped, temp = self.filter_vortex(self.v)
#        h_filtered_cropped, temp = self.filter_vortex(self.h)
#        
#        u_squared = u_filtered_cropped**2 + v_filtered_cropped**2
#        mult = self.g*h_filtered_cropped**2  + h_filtered_cropped*u_squared/2 
#        Fx = u_filtered_cropped*mult
#        Fy = v_filtered_cropped*mult
#        flux = np.array([Fx, Fy])
#        return flux
#    
#    def flux_magnitude(self):
#        flux = self.flux_vector()
#        return np.linalg.norm(flux, axis= 0)
#    
#    def flux_wave_averaged(self,):
#        fx, fy = self.flux_vector()
#        t_index = 0
#        wave_period = np.pi*2/self.omega
#        t_length = int(wave_period/(self.dt)) # this shouold be an input
#        print ('wave_period from flux wave_averaged:   ', wave_period,t_index, t_length, (self.dt) )
#        fx = fx[t_index: t_index + t_length + 0 , :, :]
#        fy = fy[t_index: t_index + t_length + 0]
#        fx_average = np.trapz(fx, dx = self.dt,  axis = 0)
#        fy_average = np.trapz(fy, dx = self.dt,  axis = 0)
#        
#        return np.array([fx_average, fy_average])/(t_length*self.dt)
    
#    def flux_wave_averaged_mag(self):
#        return np.linalg.norm(self.flux_wave_averaged(), axis = 0)
#    
#    def flux_difference(self):
#        flux_mag = self.flux_wave_averaged_mag(self.omega)
#        
#        diff = np.max(flux_mag) - np.min(flux_mag)
#        flux_incoming = 1 # palce holder
#        return diff/flux_incoming
#    
#    def flux_angle(self):
#        return None
    
    
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
        dt_fine = wave_period/N
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
        
        flux_diff = (np.max(Fm) - np.min(Fm))/Fm[int((x2-x1)/2),int((x2-x1)/5) ]
#        print (2)
#        print( np.min(Fm), np.max(Fm), Fm[int((x2-x1)/2),int((x2-x1)/5)])
        return fx_average, fy_average, flux_diff
    
    def flux_full_averaged(self):
        
                
        geoData = h5py.File(self.exp_dir + 'IC/' + 'expanded_' + self.vortex_name , 'r')

        hv = geoData.get('geoH')
        uv = geoData.get('geoU')
        vv = geoData.get('geoV')
        
        u_sub = np.subtract(self.u, uv)
        v_sub = np.subtract(self.v, vv)
        h_sub = np.subtract(self.h, hv) + self.H
        
        plt.imshow(h_sub[-1])
        plt.colorbar()
        plt.show()
        
        t_passed_vortex = self.Ly/self.c_rot*0.9
        t1 = self.closest_ind(self.time_axis, t_passed_vortex)
        wave_period = np.pi*2/self.omega
        wave_average = wave_period*2
        
        t2 = self.closest_ind(self.time_axis, t_passed_vortex + wave_average) +1
        
        #print (np.max(self.x_axis), self. Ly/2, self.wavelength)
#        zoom_index=  200
#        x1 =  zoom_index  #self.closest_ind(self.x_axis , -self.Lx/2 + self.wavelength*2)
#        x2 =  self.nx - zoom_index  #self.closest_ind(self.x_axis , self.Lx/2 - self.wavelength*2)
        
        
        x1 = self.closest_ind(self.x_axis , -self.Ly/2 + self.wavelength)
        x2 = self.closest_ind(self.x_axis , self.Ly/2 - self.wavelength)
        
        # cropped fields 
        u_crop = u_sub[t1:t2, x1:x2, x1:x2]
        v_crop = v_sub[t1:t2, x1:x2, x1:x2]
        h_crop = h_sub[t1:t2, x1:x2, x1:x2] 
        t_crop = self.time_axis[t1:t2]
        

        u_squared = u_crop**2 + v_crop**2
        mult = self.g*h_crop**2  + h_crop*u_squared/2 
        Fx = u_crop*mult
        Fy = v_crop*mult
        
        
        
        fx_average = np.trapz(Fx, dx = self.dt,  axis = 0)/wave_average
        fy_average = np.trapz(Fy, dx = self.dt,  axis = 0)/wave_average
        


        
        return np.real(fx_average), np.real(fy_average)
    
    
    def filter_flux(self):
        
        t_passed_vortex = self.Ly/self.c_rot*0.9
        t_passed_vortex_index, t_passed_closest_val = min(enumerate(self.time_axis), 
                                  key=lambda x: abs(x[1]-t_passed_vortex))
        print (t_passed_vortex_index, 't_passed vortex index HERE')
        
        cropped_omega_array =  fft.fftshift(fft.fftfreq(self.nt - t_passed_vortex_index, d=self.dt)*2 *np.pi)
        
        omega1_ind, closest_val = min(enumerate(cropped_omega_array), 
                                  key=lambda x: abs(x[1]+self.omega/3.))
        
        omega2_ind, closest_val = min(enumerate(cropped_omega_array), 
                                  key=lambda x: abs(x[1]-self.omega/3.))
        
        
        u_crop = self.u[t_passed_vortex_index:, :, :]
        u_fft =fft.fftshift(fft.fft(u_crop, axis = 0), axes=(0,))
        v_crop = self.u[t_passed_vortex_index:, :, :]
        v_fft =fft.fftshift(fft.fft(v_crop, axis = 0), axes=(0,))
        h_crop = self.u[t_passed_vortex_index:, :, :]
        h_fft =fft.fftshift(fft.fft(h_crop, axis = 0), axes=(0,))
        
        u_fft[omega1_ind:omega2_ind, :, :] = 0.0
        v_fft[omega1_ind:omega2_ind, :, :] = 0.0
        h_fft[omega1_ind:omega2_ind, :, :] = 0.0

        
        uw = np.real(fft.ifft(fft.ifftshift(u_fft, axes=(0,)), axis = 0))
        vw = np.real(fft.ifft(fft.ifftshift(v_fft, axes=(0,)), axis = 0))
        hw = np.real(fft.ifft(fft.ifftshift(h_fft, axes=(0,)), axis = 0))
        
        t2 = self.closest_ind(self.time_axis, t_passed_vortex + np.pi*2/self.omega) +1

        
        
        Fx, Fy = self.flux(uw, vw, hw)
        fx_average = np.trapz(Fx[:t2, :, :], dx = self.dt,  axis = 0)
        fy_average = np.trapz(Fy[:t2, :, :], dx = self.dt,  axis = 0)
        
        return fx_average, fy_average
#    
#    
#    def flux_omega_averaged_2(self):
#        t_passed_vortex = self.Ly/self.c_rot*1 # possible error if wave hasnt reached full domain
#        t1 = self.closest_ind(self.time_axis, t_passed_vortex)
#        wave_period = np.pi*2/self.omega
#        wave_average = wave_period*2
#        
#        t2 = self.closest_ind(self.time_axis, t_passed_vortex + wave_average) + 1#possible error if sim time isntl longe enougy
#        
#        #print (np.max(self.x_axis), self. Ly/2, self.wavelength)
##        zoom_index=  200
##        x1 =  zoom_index  #self.closest_ind(self.x_axis , -self.Lx/2 + self.wavelength*2)
##        x2 =  self.nx - zoom_index  #self.closest_ind(self.x_axis , self.Lx/2 - self.wavelength*2)
#        
#        
#        x1 =  self.closest_ind(self.x_axis , -self.Ly/2 + self.wavelength)
#        x2 = self.closest_ind(self.x_axis , self.Ly/2 - self.wavelength)
#        
#        
#
#
#        u_crop = self.u[t1:t2]
#        v_crop = self.v[t1:t2]
#        h_crop = self.h[t1:t2] 
#        t_crop = self.time_axis[t1:t2]
#        
#        yp, xp = 256, 100
#        
#        plt.scatter(t_crop, h_crop[:, yp, xp])
#        plt.plot(self.time_axis, self.h[:, yp, xp])
#        plt.title('wave averaged points used')
#        plt.show()
#        
#        omega_array =  fft.fftshift(fft.fftfreq(len(t_crop), d=self.dt)*2 *np.pi)
#        omega_ind = self.closest_ind(omega_array, self.omega)
#        omega_ind_2 = self.closest_ind(omega_array, -self.omega)
#        
#        u_fft_time = fft.fftshift(fft.fft(u_crop, axis = 0), axes=(0,))
#        v_fft_time = fft.fftshift(fft.fft(v_crop, axis = 0), axes=(0,))
#        h_fft_time = fft.fftshift(fft.fft(h_crop, axis = 0), axes=(0,))
#        
#        u_omega = np.zeros(np.shape(u_fft_time), dtype = complex)
#        v_omega = np.zeros(np.shape(v_fft_time), dtype = complex)
#        h_omega = np.zeros(np.shape(h_fft_time), dtype = complex)
#        
#        
##        zero_freq = self.closest_ind(omega_array, 0)
###        
###        u_fft_time[zero_freq] = 0.0
###        v_fft_time[zero_freq] = 0.0
###        h_fft_time[zero_freq] = 0.0
###        
##        u_omega = u_fft_time
##        v_omega = v_fft_time
##        h_omega = h_fft_time
##        
##        
##        
#        plt.scatter(omega_array/self.omega, h_fft_time[:, yp, xp])
#        plt.title('fourier transform')
#        plt.show()
#        
#        u_omega[omega_ind] = u_fft_time[omega_ind]
#        v_omega[omega_ind] = v_fft_time[omega_ind]
#        h_omega[omega_ind] = h_fft_time[omega_ind]
#        u_omega[omega_ind_2] = u_fft_time[omega_ind_2]
#        v_omega[omega_ind_2] = v_fft_time[omega_ind_2]
#        h_omega[omega_ind_2] = h_fft_time[omega_ind_2]
#        
#        uw = np.real(fft.ifft(fft.ifftshift(u_omega, axes=(0,)), axis = 0))
#        vw = np.real(fft.ifft(fft.ifftshift(v_omega, axes=(0,)), axis = 0))
#        hw = np.real(fft.ifft(fft.ifftshift(h_omega, axes=(0,)), axis = 0)) +self.H
#        
#        plt.scatter(t_crop, hw[:, yp, xp])
#        plt.scatter(t_crop, h_crop[:, yp, xp])
#        plt.title('wave vs crop')
#        plt.show()
#        
#        Fx, Fy = self.flux(uw, vw, hw)
#        
##        plt.imshow(Fx[-1])
##        plt.title('Fx ')
##        plt.colorbar()
##        plt.show()
##        
##        plt.imshow(Fy[-1])
##        plt.title('Fy ')
##        plt.colorbar()
##        plt.show()
#        
#        fx_average = np.trapz(Fx, dx = self.dt,  axis = 0)/wave_average
#        fy_average = np.trapz(Fy, dx = self.dt,  axis = 0)/wave_average
#        
#        return fx_average, fy_average    
#    
#    def flux_difference_omega(self):
#        fx, fy = self.flux_omega_averaged()
#        flux_mag = np.sqrt(fx**2 + fy**2 )
#        
#        #crop 
#
#        print (np.max(self.x_axis), self. Ly/2, self.wavelength)
#        x1 = self.closest_ind(self.x_axis , -self.Ly/2 + self.wavelength*10)
#        x2 = self.closest_ind(self.x_axis , self.Ly/2 - self.wavelength*10)
#        
#        print ( x1, x2)
#        flux_mag_cropped = flux_mag[x1:x2, x1:x2]
#        
#        plt.imshow(flux_mag_cropped/flux_mag[100,100])
#        plt.colorbar()
#        plt.show()
#        
#        diff = np.max(flux_mag_cropped) - np.min(flux_mag_cropped)
#        flux_incoming = flux_mag[0,0]
#        return diff/flux_incoming

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
        
        Ei = np.abs(E_space[256, 276])
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
        print (self.l, l_array[0], l_array[-1])
        
#        plt.imshow(np.abs(E_space))
####        plt.scatter(xp, yp)
#        plt.show()        
###        
##                

        valp = f_int(xp, yp)
        valp = []
        
        for xi, yi in zip(xp, yp):
            valp.append(f_int(xi, yi)[0])    
            

        valp = np.array(valp)
        
        max_left_lobe = np.max(valp[0: ntheta//2 +1])
#        print (valp[ntheta//2 + 1:], 'max val p')
        max_right_lobe = np.max(valp[ntheta//2 + 1:])
        
        max_ind_left = np.where(valp == max_left_lobe)[0][0]
        max_ind_right = np.where(valp == max_right_lobe)[0][0]
#        print (max_ind_left)
        #max_ind = np.where(valp == np.max(valp))[0][0]
        

#        print (theta_array[max_ind], np.max(valp)/Ei,  max_ind)
#        plt.title(self.exp_name)
#        plt.scatter(theta_array, valp)
#        plt.show()

        
        return np.max(valp)/Ei , theta_array[max_ind_left], theta_array[max_ind_right]




    def group_speed(self):
        denom = (1 + self.Lr**2*self.Bu*4*np.pi**2)**0.5
        numer = self.f*self.L*np.pi*2*self.Lr*self.Bu
        return numer/denom
        











        