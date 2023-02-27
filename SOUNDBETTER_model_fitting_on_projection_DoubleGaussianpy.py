# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:17:04 2023

@author: thoma
"""


#%% Fit a double-gaussian distribution just for SNR = -9dB.

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

realSubIDs = seedSOUNDBETTER.SubIDs

file_name = os.path.join('Subject_All_Active',
                         'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
classif = scipy.io.loadmat(file_name)
preds, conds = classif['preds'], classif['conds']
# for conds 0: audibility, 1: blocknumber, 2: evel, 3: vowel, 4: snr, 5: respside

SNRs = np.array([-20,-13,-11,-9,-7,-5])

ind_to_snr = {i+1:SNRs[i] for i in range(6)}

model_version = '_Bayesian'

#%% Create an array with the (snr*projcted activity) distributions for each subject

empirical_distributions_500ms = {}

for SubID in range(20) :
    
    print('\n ... begin with Subject'+realSubIDs[SubID]+' ...\n')
    
    # initialize variables containing the measurements of interest
    empirical_distribution_this_sub_500ms = {x:[] for x in SNRs}
    
    sfreq = 500
    
    # select the data for just this subject
    preds_this_sub_before_filtering = preds[0][SubID]
    
    # lowpass filter the data
    def butter_lowpass_filter(data, cutoff, fs, order):
        nyq = 0.5*fs
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return(y)
    preds_this_sub = butter_lowpass_filter(preds_this_sub_before_filtering,
                                           cutoff=10, fs=sfreq, order=12)
    
    # get the metadata
    snr_array = conds[0][SubID][:,4]
    
    # fill in the empirical distribution dicts WITH average on 30ms (15*2ms)
    for epo_ind, epo in enumerate(preds_this_sub):
        
        snr = ind_to_snr[snr_array[epo_ind]]
        
        # post-onset distribution
        average_value = np.mean(epo[int(1*sfreq)-8:int(1*sfreq)+7])
        empirical_distribution_this_sub_500ms[snr].append(average_value)
        
    # add this distribution to the main dict
    empirical_distributions_500ms[SubID] = empirical_distribution_this_sub_500ms
        
    

#%% Find the mean and std of the baseline activity for each subject

print('\n ... fits the noise model ...\n')

noise_parameters = {}

for SubID in range(20) :
    
    baseline_values = list(empirical_distributions_500ms[SubID][-20])
                           
    m = np.mean(baseline_values)
    s = np.std(baseline_values)
    
    noise_parameters[SubID] = (m,s)


#%% Maximize the likelihoods and store the corresponding parameters

results = {}
len_data_per_sub = {}

parameters_names = ['dist','mu_low','std_high','std_low','beta']

guesses_per_sub = {}

# Bayesian prior on beta
mu_beta, std_beta = 0.5, 0.5
mu_dist, std_dist = 2, 2

for SubID in range(20) :
    
    print('\n ... starts fit for Subject'+realSubIDs[SubID]+' ...\n')
    
    # pre - function to minimize
    def pre_MLE_bifurcation(x,parameters):
        # extract parameters
        dist, mu_low, std_high, std_low, beta = parameters
        mu_high = mu_low + dist
        
        # compute the log_likelihood
        neg_LL = 0
        for activity in x:
            p_high = np.exp(-(activity-mu_high)**2/(2*std_high**2))/(std_high*np.sqrt(2*np.pi))
            p_low = np.exp(-(activity-mu_low)**2/(2*std_low**2))/(std_low*np.sqrt(2*np.pi))
            # prior
            if model_version == '_noPrior':
                prior_beta, prior_dist = 1, 1
            if model_version == '_Bayesian':
                prior_beta = np.exp(-(beta-mu_beta)**2/(2*std_beta**2))/(std_beta*np.sqrt(2*np.pi))
                prior_dist = np.exp(-(dist-mu_dist)**2/(2*std_dist**2))/(std_dist*np.sqrt(2*np.pi))
            # likelihood
            p = beta*p_high + (1-beta)*p_low
            neg_LL -= (np.log(p) + np.log(prior_beta) + np.log(prior_dist))
            
        return(neg_LL)
    
    # definition of x (the values to fit)
    x = empirical_distributions_500ms[SubID][-9]
        
    # function to minimize
    def MLE_bifurcation(parameters):
        neg_LL = pre_MLE_bifurcation(x,parameters)
        return(neg_LL)
    
    
    # define the intial guesses for the parameters
    mu_guess = np.mean(empirical_distributions_500ms[SubID][-9])
    std_guess = np.std(empirical_distributions_500ms[SubID][-9])
    std_high_guess, std_low_guess = std_guess, std_guess
    dist_guess = 4 # mu_guess + std_guess
    mu_low_guess = -2 # mu_guess - std_guess
    beta_guess = 0.5
    guesses_per_sub[SubID] = np.array([dist_guess,mu_low_guess,std_high_guess,std_low_guess,beta_guess])
    
    
    # minimize arguments: function, intial_guess_of_parameters, method
    inf = np.inf
    bounds = scipy.optimize.Bounds(lb=[-inf,-inf,-inf,-inf,-inf],
                                   ub=[inf,inf,inf,inf,inf],
                                   keep_feasible = True)
    mle_model = minimize(MLE_bifurcation, 
                         guesses_per_sub[SubID], 
                         method='Nelder-Mead', bounds = bounds,
                         options = {'return_all':True,'maxfev':2000,'maxiter':2000})
    
    # save the obtained parameters
    results[SubID] = mle_model
    len_data_per_sub[SubID] = len(empirical_distributions_500ms[SubID][-9])
    
    # plot the evolution of the estimated parameters
    convergence = 'SUCCESSFULL'
    if not mle_model['success']:
        convergence = 'FAILURE'
    vecs = np.array(mle_model['allvecs'])
    plt.figure()
    plt.title('Subject'+realSubIDs[SubID]+' ('+convergence+')')
    for i in range(5):
        plt.plot(vecs[:,i],label=parameters_names[i])
    plt.legend()
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')
    plt.savefig('Parameters_Convergence_Subject'+realSubIDs[SubID]+model_version+'_DoubleGaussian')
    os.chdir(datapath)
    plt.close()
    
    print('\n... convergence for Subject'+realSubIDs[SubID]+' : ', convergence, '\n result : \n')
    for i in range(len(parameters_names)):
        print(parameters_names[i] + ' = ',mle_model.x[i])


print('\n... save the results ... \n')

# save the results
for SubID in range(20):
    
    # get the resulting model for this subject
    model = results[SubID].copy()
    
    # remove what we don't want, add what we want
    model['average_likelihood'] = np.exp(-model['fun']/len_data_per_sub[SubID])
    del model['final_simplex'], model['allvecs'], model['message']
    x = model.pop('x')
    for ipara, para_name in enumerate(parameters_names):
        model[para_name] = x[ipara]
    mu_noise, std_noise = noise_parameters[SubID]
    model['mu_noise'], model['std_noise'] = mu_noise, std_noise
        
    # save model as a DataFrame
    for key in model.keys():
        model[key] = [model[key]]
    model_to_save = pandas.DataFrame(model)
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')
    model_to_save.to_csv('ModelFit_Subject'+realSubIDs[SubID]+model_version+'_DoubleGaussian')
    os.chdir(datapath)

#%% Plot empirical and fit distributions superposed to evaluate the fit

for SubID in range(20) : 
    
    print('\n ... distribution plot for Subject'+realSubIDs[SubID]+' ...\n')
    
    # get model fit arguments
    dist, mu_low, std_high, std_low, beta = results[SubID]['x']
    mu_high = mu_low + dist
    
    convergence = 'SUCCESSFULL'
    if not results[SubID]['success']:
        convergence = 'FAILURE'
    
    # get fit lists for high and low states
    activities = np.linspace(-3.5,3.5,100)
    high_probas = []
    low_probas = []
    for activity in activities:
        p_high = np.exp(-(activity-mu_high)**2/(2*std_high**2))/(std_high*np.sqrt(2*np.pi))
        p_low = np.exp(-(activity-mu_low)**2/(2*std_low**2))/(std_low*np.sqrt(2*np.pi))
        high_probas.append(p_high*beta)
        low_probas.append(p_low*(1-beta))
    
    # plot
    plt.figure()
    plt.title('Empirical and fit distributions for SubJect'+realSubIDs[SubID]+' ('+convergence+')')
    real_means = [np.mean(empirical_distributions_500ms[SubID][-9])]
    plt.hist(empirical_distributions_500ms[SubID][-9],bins=30,density=True,range=(-3.5,3.5))
    plt.plot(activities,low_probas,color='red',label='low state')
    plt.plot(activities,high_probas,color='green',label='high state p='+str(int(beta*1000)/1000))
    plt.legend()
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')
    plt.savefig('Empirical_VS_Fit_Subject'+realSubIDs[SubID]+model_version+'_DoubleGaussian')
    os.chdir(datapath)
    plt.close()