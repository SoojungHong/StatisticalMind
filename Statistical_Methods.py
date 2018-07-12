#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:36:30 2018

@author: soojunghong

@about : experiment to evaluate the model using metrics 
"""

#------------------------------------
# Box-Cox Power transformation
#------------------------------------
from scipy import stats
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(211)
x = stats.loggamma.rvs(5, size=500) + 5
prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

# use box cox 
ax2 = fig.add_subplot(212)
xt, _ = stats.boxcox(x)
prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')

plt.show()


#--------------
# t-test 
#--------------
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

import numpy as np
import pandas as pd
import scipy

# some normal distribution with (mean = 0, variance 1)
data1 = np.random.normal(0, 1, size=50)
data2 = np.random.normal(2, 1, size=50)

x = np.linspace(-4, 4, 160)
y1 = scipy.stats.norm.pdf(x)
y2 = scipy.stats.norm.pdf(x, loc=2)

trace1 = go.Scatter(
    x = x,
    y = y1,
    mode = 'lines+markers',
    name='Mean of 0'
)

trace2 = go.Scatter(
    x = x,
    y = y2,
    mode = 'lines+markers',
    name='Mean of 2'
)

data = [trace1, trace2]

py.iplot(data, filename='normal-dists-plot')


#-----------------
# T-test example
#-----------------
from pandas import DataFrame

data = {'Category': ['cat2','cat1','cat2','cat1','cat2','cat1','cat2','cat1','cat1','cat1','cat2'],
        'values': [1,2,3,1,2,3,1,2,3,5,1]}

my_data = DataFrame(data)
my_data.groupby('Category').mean()

from scipy.stats import ttest_ind

cat1 = my_data[my_data['Category'] == 'cat1']
cat2 = my_data[my_data['Category'] == 'cat2']
ttest_ind(cat1['values'], cat2['values']) # ttest_ind : Calculate the T-test for the means of two independent samples of scores.



#---------------------------------------------------------------------------------------------------------------------------------------------
# scipy.stats.ttest_ind() is to calculate the T-test for the means of Two independent samples of scores
# 
# with real world example. 
# null-hypothesis : there is no statistically significant difference in the mean of male consulting doctor and junior resident female doctors
#---------------------------------------------------------------------------------------------------------------------------------------------
female_doctor_bps = [128, 127, 118, 115, 144, 142, 133, 140, 132, 131, 
                     111, 132, 149, 122, 139, 119, 136, 129, 126, 128]

male_consultant_bps = [118, 115, 112, 120, 124, 130, 123, 110, 120, 121,
                      123, 125, 129, 130, 112, 117, 119, 120, 123, 128]

stats.ttest_ind(female_doctor_bps, male_consultant_bps)
#p-value is less than 0.05, we reject the null hypothesis, means that we can say there is statistically signigicant difference betwwen two groups 


#------------------------------------------------------------------------------------------------------------
# ANOVA test - used to compare the means of three or more samples
# ANOVA will provide an F-statistics which can, along with degrees of freedom, be used to calculate p-value
#-------------------------------------------------------------------------------------------------------------
ctrl = [4.17, 5.58, 5.18, 6.11, 4.5, 4.61, 5.17, 4.53, 5.33, 5.14]
trt1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
trt2 = [6.31, 5.12, 5.54, 5.5, 5.37, 5.29, 4.92, 6.15, 5.8, 5.26]

stats.f_oneway(ctrl, trt1, trt2)


#---------------------------------------
# Data representation and interaction
#---------------------------------------
import pandas
data = pandas.read_csv('/Users/soojunghong/Documents/safariML/Statistical_Method/brain_size.csv', sep = ';', na_values = ".")
data

import numpy as np
t = np.linspace(-6, 6, 20) #from -6 until 6, 20 numbers 
t
sin_t = np.sin(t)
cos_t = np.cos(t)

pandas.DataFrame({'t':t, 'sin':sin_t, 'cos':cos_t})

data.shape #(40, 8) - 40 rows and 8 columns 
data.columns
pandas.DataFrame.describe(data) #it shows count, mean, std, min 25%, 50%, 75%
print(data['Gender'])

#groupby - splitting a dataframe on values on categorical variables 
groupby_gender = data.groupby('Gender')
groupby_gender
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))

groupby_gender.mean()  

# plotting data
from pandas.tools import plotting
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])  