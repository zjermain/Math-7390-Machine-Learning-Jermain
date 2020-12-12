# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:32:30 2020

@author: zjerma1
"""




import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from mlxtend.evaluate import bias_variance_decomp

#data for final plot 
x_plot = np.linspace(0,1,num=10) 
y_plot = np.sin(2*np.pi*x_plot)
j = 0
#y_avg = np.array[100]
test_error = []
bias = []
variance = []
reg_parameter = []
bias_variance = []
for i in range(8): 
    reg_parameter.append(10**i)
    reg_parameter.append(10**(-i))
reg_parameter.sort()
print(reg_parameter)

for j in reg_parameter:
    error_holder = 0
    bias_holder = 0
    variance_holder = 0
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
    ridge_reg = Ridge(alpha = j ,solver="cholesky")

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

    error_holder, bias_holder, variance_holder = bias_variance_decomp(ridge_reg, x_data, y_data, x_new, y_new, loss = 'mse') #bias-variance decomp
    test_error.append(error_holder)
    bias.append(bias_holder)
    variance.append(variance_holder)

for j in range(len(bias)): 
    bias[j] = bias[j]**2
for j in range(len(bias)): 
    bias_variance.append(bias[j] + variance[j])

print(test_error)
print(bias_variance)
 
plt.plot(reg_parameter, test_error, label = 'test error')
plt.plot(reg_parameter, bias, label = 'bias')
plt.plot(reg_parameter,variance, label = 'variance')
plt.plot(reg_parameter, bias_variance, label = 'bias + variance')
plt.xscale('log')
plt.legend()
plt.show