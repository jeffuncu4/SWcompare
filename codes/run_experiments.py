#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 09:58:02 2021

@author: jeff
"""

from simulationClass import Simulation 
import numpy as np
import matplotlib.pyplot as plt


def round_arange(start, stop, step):
    step_decimal_points = len(str(step).split('.')[-1])
    start_decimal_points = len(str(start).split('.')[-1])
    stop_decimal_points = len(str(stop).split('.')[-1])
    max_dec_points = max([start_decimal_points, stop_decimal_points, step_decimal_points])

    num_list =  np.arange(start, stop, step)
    num_rounded_list = []
    for num in num_list:
        rounded_num = round(num, max_dec_points)
        num_rounded_list.append(float(str(rounded_num).rstrip('0')))
    return np.array(num_rounded_list)


def sim_iterator(Ros, Bus, Lrs, Urs):
    # input non dim numers as tuples
    for ro in Ros:
        for bu in Bus:
            for lr in Lrs:
                for ur in Urs:
                    exp = Simulation(ro, bu, lr, ur)
                    exp.run_sim()
    return None




#
Ro = [0.01] # must make these floats for proper naming conventions
Bu = [1.]
Lr = [4.]
Ur = [1000.]


sim_iterator(Ro, Bu, Lr, Ur)

#
#
#
#






        