#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:35:13 2022

@author: jeff
"""

import numpy as np
import matplotlib.pyplot as plt


a = [1,2]
b = [1,3]
x = [1,4]
y = [1,6]

#plt.plot(a, b, label = 'Ro = 0.02')
#plt.plot(a, x, label = 'Ro = 0.04')
#plt.plot(a, y, label = 'Ro = 0.08')
#plt.title('Energy conversion')
#plt.xlabel(r'$ln(Ro^{\alpha} Bu^{\beta} \Lambda^{\gamma})$')   
#plt.ylabel('Scattered/Incoming Energy')          
#plt.show()
#
#

b = 5

x = np.arange(9)
y = np.arange(9)

xx, yy = np.meshgrid(x, y)
z = xx**0.5 + yy


plt.imshow(z)
plt.colorbar()
plt.show()