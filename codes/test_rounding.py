#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 09:47:10 2021

@author: jeff
"""
import numpy as np

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

start = 0.0
stop = 4000.0
step = 1000.0


start = 0.0
stop = 1
step = 1/3.

l = round_arange(start, stop, step)
print (l)

#for i in round_arange():
#    