#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:00:16 2018

@author: soojunghong

"""

import scipy
from scipy.stats import pearsonr

x = scipy.array([-0.6549, 2.3464, 3.0])
y = scipy.array([-1.4604, 3.8653, 21.0])

r_row, p_value = pearsonr(x,y)
r_row
p_value