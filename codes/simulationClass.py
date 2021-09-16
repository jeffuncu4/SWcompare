#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 19:04:08 2021

@author: jeff
"""


import numpy as np
import os
import subprocess
import pickle
from AnalysisClass import Analysis
from PlottingClass import Plot


class Simulation:
    def __init__(self, Ro, Bu, Lr, Ur):
        ###PARAMETERS are intialized as floats 
        
        #non dimensional numbers
        self.Ro = Ro
        self.Bu  = Bu
        self.Lr = Lr
        self.Ur = Ur
        
        self.bash_script_dir = '../bash_scripts/'
        ### SIMULATION LOGISITICS
        self.exp_name = 'Ro{}Bu{}Lr{}Ur{}'.format(self.Ro, self.Bu, self.Lr, self.Ur)
        self.exp_dir = '../experiments/' + self.exp_name + '/'
                                                                  

#        self.sim_executed = 'state1.h5' in os.listdir(self.exp_dir + 'data')
#        self.analytics_executed = 'somefile' in os.listdir("experiments/{}/analysis/"
#                                                      .format(self.exp_name))
        self.analysis = None
#        
        self.plots = Plot(Ro, Bu, Lr, Ur)
        

        
        #CONSTANTS
        g = 9.81 # m/s**2 
        
        #Dimensional numbers I choose
        f = 1e-4 # 1/s
        L = 25000 #m
        
        #Resulting dimensional quantities DEFUNE HG FIRST NOT SURE IF THIS I SMART
        wavelength = L/self.Lr
        H = self.Bu*(f*L)**2/g
        
        
        
        #Variables which are functios of the non dimensioanl numbers
        l = np.pi*2/wavelength
        self.omega = np.sqrt(g*l**2*H + f**2)


    def create_sim(self, run_if_created = False):       
        sim_created = self.exp_name in os.listdir("../experiments/")
        if sim_created and not run_if_created:
            print ("Simulation already created")
            return None
        
        subprocess.call(self.bash_script_dir +"create_sim.sh " + self.exp_name,  shell=True)
            
        

        Re = 1e6
        sponge_decay = 0.05
        forcing_decay = 0.2
        
        #CONSTANTS
        g = 9.81 # m/s**2 
        
        #Dimensional numbers I choose
        f = 1e-4 # 1/s
        L = 25000 #m
        
        #Resulting dimensional quantities DEFUNE HG FIRST NOT SURE IF THIS I SMART
        wavelength = L/self.Lr
        hg = self.Ro*(L*f)**2/g
        ug  = hg/(L)*g/f # Not explicitly used in cyclogestorphic vortices
        H = self.Bu*(f*L)**2/g
        uw = ug/self.Ur
        
        
        
        #Variables which are functios of the non dimensioanl numbers
        l = np.pi*2/wavelength
        self.omega = np.sqrt(g*l**2*H + f**2) # negative for opposite propogating wave
        #hg =  ug*f/(l*g) #H*Ro # I dont think I ever use this in the code
        hw = l*H*uw/f
        vw = uw*self.omega/f
        tau_s = sponge_decay*2*np.pi/self.omega
        tau_w = forcing_decay*2*np.pi/self.omega
        c = np.sqrt(g*H)
        Ld = c/f
        
#        print (omega, 'omega')
        
        
        
        #Simulations  non dimensional numbers
        L_aspect = 1
        forcing_window_length_wavelength = 1.
        sponge_window_length_wavelength = 1.
        geo_window_length_ratio = 0.2 # doesnt do anyting rn
        nx = 512
        # I choose to define dx, dt for the simulation Nondim# 
        #Keep in mind I neeed to satisfy some sort of cfl condition with dt <  C*dx/U
        CFL_constant = 0.002
        
        dx = 500 # m
        dt = CFL_constant*dx/ug # seconds
         
        # The resulting parameters are
        mu = ug*dx**3/Re  #m^4/s
        # not true if you use CFL
    
        
        Lx = nx*dx
        Ly = Lx*L_aspect
        ny = nx*L_aspect
        forcing_window_length = forcing_window_length_wavelength*(np.pi*2/l)
        sponge_window_length = sponge_window_length_wavelength*(np.pi*2/l)
        geo_window_length = geo_window_length_ratio*(np.pi*2/l)
        
        
        inertial_period = np.pi*2/f
        c_rot = self.omega/l
        num_periods = Ly*4./3./c_rot/inertial_period 
        Ts = Ly/c_rot*1.3
        
        #Other Numbers which are relevant 
        T_wave = np.pi*2/self.omega        

        save_iter = 100 # main sim save iteration
        num_iter = int(Ts/dt)
        
        save_iter_vortex= 10 # I dont think this is even in use
        num_iter_vortex = int(inertial_period*.5/dt)
        
        max_writes = 100000        
        
        nx_v = 512
        Lx_v = nx_v*dx
        Ly_v = Lx_v*L_aspect
        ny_v = nx_v*L_aspect
        
        
        parameterValues = [self.Ur,self.Lr,self.Ro , self.Bu, Re,sponge_decay,forcing_decay,ug ,f,g,uw, L, l,
                H ,mu, self.omega, hg ,hw ,vw ,tau_s,tau_w ,num_periods,
                L_aspect,forcing_window_length_wavelength,sponge_window_length_wavelength ,
                geo_window_length_ratio ,dt,dx, Ts ,num_iter, Lx ,Ly ,nx,
                forcing_window_length,sponge_window_length,geo_window_length ,
                save_iter ,ny ,T_wave,wavelength,c ,Ld ,max_writes, num_iter_vortex, save_iter_vortex, nx_v, Lx_v, Ly_v, ny_v ]


        parameterNames = ['Ur','Lr','Ro' ,'Bu' ,'Re','sponge_decay','forcing_decay','ug' ,'f','g','uw', 'L', 'l',
                'H' ,'mu', 'omega','hg' ,'hw' ,'vw' ,'tau_s','tau_w' ,'num_periods',
                'L_aspect','forcing_window_length_wavelength','sponge_window_length_wavelength' ,
                'geo_window_length_ratio' ,'dt','dx', 'Ts' ,'num_iter', 'Lx' ,'Ly' ,'nx',
                'forcing_window_length','sponge_window_length','geo_window_length' ,
                'save_iter' ,'ny' ,'T_wave','wavelength','c' ,'Ld' ,'max_writes',
                'num_iter_vortex', 'save_iter_vortex',  'nx_v', 'Lx_v', 'Ly_v', 'ny_v' ]
        
        my_dict = dict(zip(parameterNames, parameterValues))
        with open('../experiments/{}/IC/parameters.pkl'.format(self.exp_name), 
                  'wb') as pfile:
            pickle.dump(my_dict, pfile)
            
        
            
        subprocess.call('python forcing_windows.py ' + self.exp_name,  shell=True)

   
    def run_vortex_sim(self, run_if_created = False):
        ''' this will run the cylogeostophic djustment code as and run the simulation
        until it settles'''
        vortex_created = 'settled_vortex.h5' in os.listdir(self.exp_dir + 'IC/')
        if vortex_created and not run_if_created:
            print ("vortex already created already created")
            return None
        subprocess.call(self.bash_script_dir +"run_vortex_sim.sh {}".format(self.exp_name), shell=True)
    
        
    def run_main_sim(self, run_if_created = False): #this creates and runs the simulation
        # this should check if the siulation already exist in vortex_fields
        data_created = 'data_s1.h5' in os.listdir(self.exp_dir + 'data/')
        if data_created and not run_if_created:
            print ("main sim has been run")
            self.analysis = Analysis(self.Ro, self.Bu, self.Lr, self.Ur)
            return None
        subprocess.call(self.bash_script_dir + "run_main_sim.sh {}".format(self.exp_name), shell=True)
        self.analysis = Analysis(self.Ro, self.Bu, self.Lr, self.Ur)
    

    def make_uvh_videos(self, run_if_created=False):
        vids_created = 'h.mp4' in os.listdir(self.exp_dir + 'plots/')
        if vids_created and not run_if_created:
            print ("uvh videos have been created")
            return None
        subprocess.call(self.bash_script_dir + "make_uvh_videos.sh {}".format(self.exp_name), shell=True)
    
    def run_analysis(self):
        pass

    def run_sim(self, run_if_created=False):
        self.create_sim(run_if_created)
        self.run_vortex_sim(run_if_created)
        self.run_main_sim(run_if_created)
        self.make_uvh_videos(run_if_created)


#
#
#    def run_and_analyze(self):
#        self.run_all()
#        self.make_videos()
#        self.run_analysis()


#    def run_analytics(self):
#        if self.sim_executed == True:
#            subprocess.call("run_analytics.sh {}".format(self.expName))
#            
#    def plot_all(self):
#        if self.sim_executed == True:
#            subprocess.call("plot_all.sh {}".format(self.expName))
#        
