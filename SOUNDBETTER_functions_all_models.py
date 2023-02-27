# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:37:59 2023

@author: thoma
"""


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
import scipy
import pandas
from scipy.signal import butter,filtfilt

datapath = seedSOUNDBETTER.datapath
os.chdir(datapath)

mat2mne = seedSOUNDBETTER.mat2mne



realSubIDs = seedSOUNDBETTER.SubIDs

file_name = os.path.join('Subject_All_Active',
                         'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
classif = scipy.io.loadmat(file_name)
preds, conds = classif['preds'], classif['conds']
# for conds 0: audibility, 1: blocknumber, 2: evel, 3: vowel, 4: snr, 5: respside

SNRs = np.array([-20,-13,-11,-9,-7,-5])

ind_to_snr = {i+1:SNRs[i] for i in range(6)}


EmpiricalDistribution = seedSOUNDBETTER.EmpiricalDistribution



#%% _Bayesian00

def fit_Bayesian00(data,iparameters,priors,maxfev=2000,maxiter=2000):
    
    '''
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    iparameters : len 7 list or array of the initial guesses 
        (['k','threshsnr','step','L_high','k_high','mu_noise','std_noise']).
    priors : len 7 list or array of (mu_param, std_param) tuples or None (if no prior to apply).
    maxfev, maxiter : int, optionnal. Max number of function evaluation / iterations. Default 2000.

    Returns
    -------
    success : bool informing about the convergence of the optimization.
    parameters : dict of the final parameters obtained.
    fun : float, final value of p(this_model|data) (or p(data|this_model) if no prior).
    
    '''
    
    # unpack the initial parameters
    k_guess, threshsnr_guess, step_guess, L_high_guess, k_high_guess, mu_noise_guess, std_noise_guess = iparameters
    
    # pre - function to minimize
    def pre_MLE_bifurcation(data,parameters):
        # extract parameters
        k, threshsnr, step, L_high, k_high, mu_noise, std_noise = parameters
        
        # compute the log_likelihood
        neg_LL = 0
        for snr in SNRs :
            for ind, activity in enumerate(data[snr]):
                mu_snr = L_high/(1+np.exp(-k_high*(snr-threshsnr))) + step
                p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
                p_low = np.exp(-(activity-mu_noise)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
                beta = 1/(1+np.exp(-k*(snr-threshsnr)))
                # likelihood
                p_likelihood = (beta*p_high + (1-beta)*p_low)
                neg_LL -= np.log(p_likelihood)
                # prior probabilities
                for iparam, prior in enumerate(priors) :
                    if prior != None :
                        param = parameters[iparam]
                        mu_param, std_param = prior
                        prior = np.exp(-(param-mu_param)**2/(2*std_param**2))/(std_param*np.sqrt(2*np.pi))
                        neg_LL -= np.log(prior) 
        return(neg_LL)
    
    # function to minimize
    def MLE_bifurcation(parameters):
        neg_LL = pre_MLE_bifurcation(data,parameters)
        return(neg_LL)
    
    # minimize fun in the parameters space
    mle_model = minimize(MLE_bifurcation, 
                         np.array([k_guess, threshsnr_guess, step_guess, L_high_guess, k_high_guess, mu_noise_guess, std_noise_guess]), 
                         method='Nelder-Mead',
                         options = {'return_all':False,'maxfev':maxfev,'maxiter':maxiter})
    
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    return(mle_model['success'], mle_model['x'], mle_model['fun'], nb_points)


def test_Bayesian00(data,parameters):
    
    '''
    
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 7 list or array of already fitted parameters.

    Returns
    -------
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    k, threshsnr, step, L_high, k_high, mu_noise, std_noise = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for snr in SNRs :
        for ind, activity in enumerate(data[snr]):
            mu_snr = L_high/(1+np.exp(-k_high*(snr-threshsnr))) + step
            p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            p_low = np.exp(-(activity-mu_noise)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            beta = 1/(1+np.exp(-k*(snr-threshsnr)))
            # likelihood
            p_likelihood = (beta*p_high + (1-beta)*p_low)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(average_likelihood)
    