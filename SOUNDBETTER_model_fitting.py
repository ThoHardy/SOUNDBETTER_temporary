# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:12:27 2023

@author: thoma
"""

#%% Fit the Bifurcation model with the data, pre-selecting as much parameters 
# as possible before using the MLE method.

#%% Import toolboxes and functions, and def 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api
from scipy import stats
from scipy.optimize import minimize 
import seedSOUNDBETTER
import mne
import os

datapath = seedSOUNDBETTER.datapath
mat2mne = seedSOUNDBETTER.mat2mne

SubIDs = seedSOUNDBETTER.SubIDs

#%% Fit the model for each trial and keeps the result in a list

X = np.array([-13,-11,-9,-7,-5])

parameters = []

for SubID in SubIDs :
    
    file_name = os.path.join('Subject_' + SubID + '_Active','data_ref.mat')
    
    # load epochs, get data
    epochs, y = mat2mne(datapath + file_name,1)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    
    # demean
    mean_baseline = np.mean(data[:,:,:int(0.5*sfreq)],2)
    data = np.array([[data[i][j] - mean_baseline[i][j] for j in range(len(data[0]))] for i in range(len(data))])
    
    # transform into the projected value
        # none for the moment.

    #mean_minsnr = np.mean(epochs)

    # fit for each event
    for epo_ind, epo in enumerate(data):
        
        # noise distribution (its mean is 0 because of the demean)
        baseline_data = epo[:,:int(0.5*sfreq)]
        mean_noise = np.mean(data[:,:,:int(0.5*sfreq)],1) # for now, obviously 0
        sigma_noise = np.std(baseline_data,1)
        
        step = mean_minsnr - mean_noise
        
        def MLE_Norm(parameters):
          # extract parameters
          const, beta, std_dev = parameters
          # predict the output
          pred = const + beta*X
          # Calculate the log-likelihood for normal distribution
          LL = np.sum(stats.norm.logpdf(y, pred, std_dev))
          # Calculate the negative log-likelihood
          neg_LL = -1*LL
          return neg_LL
