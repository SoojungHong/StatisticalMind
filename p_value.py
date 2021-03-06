#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:52:39 2018

@reference : https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f

"""

#-----------------------
## Import the packages
#-----------------------
import numpy as np
from scipy import stats


#---------------------------------
## Define 2 random distributions
#---------------------------------
#Sample Size
N = 10
#Gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N) + 2
#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N) # return narray
a
b

#-------------------------------------------------------
## Calculate the Standard Deviation
# Calculate the variance to get the standard deviation
#-------------------------------------------------------

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a = a.var(ddof=1) # var() is Compute the variance along the specified axis.
var_b = b.var(ddof=1)

var_a
var_b

#------------------
# std deviation
#------------------
s = np.sqrt((var_a + var_b)/2)
s


#---------------------------------
## Calculate the t-statistics
#---------------------------------
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))


#--------------------------------------
## Compare with the critical t-value
# Degrees of freedom
#--------------------------------------
df = 2*N - 2

#--------------------------------------
# p-value after comparison with the t 
#--------------------------------------
p = 1 - stats.t.cdf(t,df=df) #cdf(x, df, loc=0, scale=1) 	Cumulative density function


print("t = " + str(t))
print("p = " + str(2*p))
#Note that we multiply the p value by 2 because its a twp tail t-test
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.