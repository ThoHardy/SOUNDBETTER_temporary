# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:12:27 2023

@author: thoma
"""

#%% Fit the Bifurcation model with the data, pre-selecting as the noise mean and std 
# before using the MLE method.

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

SNRs = np.array([-20,-13,-11,-9,-7,-5])

ind_to_snr = {i+1:SNRs[i] for i in range(6)}



#%% Create an array with the (snr*activity) distributions for each subject

empirical_distributions_500ms = {}

for SubID in SubIDs :
    
    # initialize variables containing the measurements of interest
    empirical_distribution_this_sub_500ms = {x:[] for x in SNRs}
    
    file_name = os.path.join('Subject_' + SubID + '_Active','data_ref.mat')
    
    # load epochs, get data (just from Cz for the moment)
    epochs, y = mat2mne(datapath + file_name,1)
    sfreq = epochs.info['sfreq']
    data = np.array([epo[0] for epo in epochs.pick('Cz').get_data()])
    
    # demean
    mean_baseline = np.mean(data[:,:int(0.5*sfreq)],1)
    data = np.array([data[i] - mean_baseline[i] for i in range(len(data))])
    
    # transform into the projected value
        # none for the moment.

    # fill in the empirical distribution dicts
    for epo_ind, epo in enumerate(data):
        
        snr = ind_to_snr[y['snr'][epo_ind]]
        
        # post-onset distribution
        empirical_distribution_this_sub_500ms[snr].append(epo[int(1*sfreq)])
        
    # add this distribution to the main dict
    empirical_distributions_500ms[SubID] = empirical_distribution_this_sub_500ms
        
    

#%% Find the mean and std of the baseline activity for each subject

noise_parameters = {}

for SubID in SubIDs :
    
    baseline_values = list(empirical_distributions_500ms[SubID][-20])
                           
    m = np.mean(baseline_values)
    s = np.std(baseline_values)
    
    noise_parameters[SubID] = (m,s)




#%% Maximize the likelihoods and store the corresponding parameters

parameters = {}

for SubID in SubIDs :
    
    # get noise parameters 
    mu_noise, std_noise = noise_parameters[SubID]
    
    # pre - function to minimize
    def pre_MLE_bifurcation(x,parameters):
        # extract parameters
        k, threshsnr, step, L_high, k_high = parameters
        # len of the data for 1 SNR
        l = len(x)//len(SNRs)
        
        # compute the log_likelihood
        neg_LL = 0
        for ind, activity in enumerate(x):
            snr = SNRs[ind//l]
            mu_snr = L_high/(1+np.exp(-k_high*(snr-threshsnr))) + step
            p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            p_low = np.exp(-(activity-mu_noise)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            beta = 1/(1+np.exp(-k*(snr-threshsnr)))
            # p(x=activity|SNR=snr)
            p = beta*p_high + (1-beta)*p_low
            neg_LL -= np.log(p)
            
        return(neg_LL)
    
    # definition of x (the values to fit)
    min_len = np.min([len(empirical_distributions_500ms[SubID][k]) for k in empirical_distributions_500ms[SubID].keys()])
    x = [] # x will contain exactly min_len values for each snr
    for k in empirical_distributions_500ms[SubID].keys():
        x = x + empirical_distributions_500ms[SubID][k][:min_len]
        
    # function to minimize
    def MLE_bifurcation(parameters):
        neg_LL = pre_MLE_bifurcation(x,parameters)
        return(neg_LL)
    
    
    # define the intial guesses for the parameters
    L_high_guess = np.mean(empirical_distributions_500ms[SubID][-5]) - np.mean(empirical_distributions_500ms[SubID][-13])
    k_guess, k_high_guess = L_high_guess/8, L_high_guess/8
    step_guess = np.mean(empirical_distributions_500ms[SubID][-13]) - np.mean(empirical_distributions_500ms[SubID][-20])
    threshsnr_guess = (np.mean(empirical_distributions_500ms[SubID][-5]) + np.mean(empirical_distributions_500ms[SubID][-13]))/2
    
    
    
    # minimize arguments: function, intial_guess_of_parameters, method
    mle_model = minimize(MLE_bifurcation, 
                         np.array([k_guess, threshsnr_guess, step_guess, L_high_guess, k_high_guess]), 
                         method='Nelder-Mead')
    
    # save the obtained parameters
    parameters[SubID] = mle_model.x
    
    
print(parameters)

