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
ttest_ind(cat1['values'], cat2['values'])