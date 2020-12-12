# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:14:08 2020

@author: zjermain15
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

#data for final plot 
x_plot = np.linspace(0,1,num=10) 
y_plot = np.sin(2*np.pi*x_plot)

#generate training data
x_data = np.linspace(0,1,num=10).reshape(-1,1)
y_data = np.sin(2.0*np.pi*x_plot) + .1*np.random.randn(10)
#print(x_plot)
#print("\n")
#print(y_data)

#add polynomials to the model 
poly_features = PolynomialFeatures(degree=9,include_bias=False)
#print(poly_features)

#x_data is extened by including its powers 
x_data_poly = poly_features.fit_transform(x_data)
#print(x_data_poly)

#Ridge regression or Tikhonov regularizaiton 
ridge_reg = Ridge(alpha=0.001,solver="cholesky")

#fit the extended data set 
ridge_reg.fit(x_data_poly,y_data)

#generate the test data set 
x_new = np.linspace(0,1,num = 100).reshape(100,1)
x_new_poly = poly_features.transform(x_new)
#print(x_new)
#print('\n')
#print(x_new_poly)

#prediction on the test data set 
y_new = ridge_reg.predict(x_new_poly)
#print(y_new)

print(cross_val_score(ridge_reg, x_new_poly, y_new, cv=9))
print(cross_val_score(ridge_reg, x_data_poly, y_data, cv = 5))
print(np.mean(cross_val_score(ridge_reg, x_new_poly, y_new, cv=9)))
print(np.mean(cross_val_score(ridge_reg, x_data_poly, y_data, cv = 5)))
#print(x_new_poly)
#print('\n')
#print(y_new)
#print(len(x_new_poly))
#print(len(y_new))
plt.scatter(x_plot,y_plot,label = 'sin(x)')
plt.scatter(x_data,y_data, label = 'training data')
#plt.plot(x_data_poly,y_data)
plt.plot(x_new,y_new, label = 'Ridge Regression')
#for j in range(10): 
#    plt.plot(x_new_poly[j:],y_new[j:], label = j+1)
#for i in range(len(y_new)): 
#    for j in range(9): 
#        plt.scatter(x_new_poly[i][j],y_new[i][j], label = j+1)
plt.legend()
plt.show