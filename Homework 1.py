# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:12:05 2020

@author: zjermain15
"""

import numpy as np
import matplotlib.pyplot as plt

size = [10,100,1000,10000,100000,1000000]
error = []
for N in size:
    rv = np.random.randn(N)


    values = []
     
    for j in range(N): 
        values.append(np.exp(-rv[j]**2))
    error.append(abs(sum(values)/N-(1/np.sqrt(3))))
    
print(error)
plt.scatter(size,error)
plt.xscale('log')
plt.title('error vs N')
plt.xlabel('N')
plt.ylabel('error')
plt.show()





    