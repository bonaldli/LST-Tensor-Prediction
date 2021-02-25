# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:51:14 2020

@author: zlibn
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.io


cpfactor2_train = np.zeros((56,50))
cpfactor2_train[0:52,:] = deepcopy(cpfactor2[0:52,:])

cpfactor2_test = cpfactor2[52:56,:]

R0 = np.transpose(np.reshape(cpfactor2_train[:,0], (-1, 7)))
R0_r = np.transpose(np.reshape(cpfactor2[:,0], (-1, 7)))
scipy.io.savemat('C:/Users/zlibn/Desktop/2D_ARMA/R0.mat', mdict={'R0': R0})

np.var(cpfactor2[0:52,:])






