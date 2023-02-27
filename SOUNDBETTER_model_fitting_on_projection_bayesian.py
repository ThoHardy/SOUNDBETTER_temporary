# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:49:23 2023

@author: thoma
"""



#%% Fit the Bifurcation model with the data, using only MLE on all the parameters,
# with Bayesian conditions on them 
# (L_high = mu_maxsnr +- std_noise ; step = 0 +- 1 ; threshsnr = -9 +- 1)

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

parameters_names = ['k','threshsnr','step','L_high','k_high','mu_noise','std_noise']

for SubID in range(20) :
    
    print('\n ... starts fit for Subject'+realSubIDs[SubID]+' ...\n')
    
    # priors on relevant parameters :
    mu_threshsnr, std_threshsnr = -9, 2
    mu_L_high, std_L_high = np.mean(empirical_distributions_500ms[SubID][-5]) - np.mean(empirical_distributions_500ms[SubID][-13]), 2*noise_parameters[SubID][0]
    mu_step, std_step = np.mean(empirical_distributions_500ms[SubID][-13]) - np.mean(empirical_distributions_500ms[SubID][-20]), 1
    
    # pre - function to minimize
    def pre_MLE_bifurcation(x,parameters):
        # extract parameters
        k, threshsnr, step, L_high, k_high, mu_noise, std_noise = parameters
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
            # prior probability
            prior_threshsnr = np.exp(-(threshsnr-mu_threshsnr)**2/(2*std_threshsnr**2))/(std_threshsnr*np.sqrt(2*np.pi))
            prior_L_high = np.exp(-(L_high-mu_L_high)**2/(2*std_L_high**2))/(std_L_high*np.sqrt(2*np.pi))
            prior_step = np.exp(-(step-mu_step)**2/(2*std_step**2))/(std_step*np.sqrt(2*np.pi))
            p = beta*p_high + (1-beta)*p_low
            neg_LL -= (np.log(p)) # + np.log(prior_threshsnr) +np.log(prior_step)) #np.log(prior_L_high))
            
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
    threshsnr_guess = -9
    mu_noise_guess, std_noise_guess = noise_parameters[SubID]
    
    
    
    # minimize arguments: function, intial_guess_of_parameters, method
    mle_model = minimize(MLE_bifurcation, 
                         np.array([k_guess, threshsnr_guess, step_guess, L_high_guess, k_high_guess, mu_noise_guess, std_noise_guess]), 
                         method='Nelder-Mead',
                         options = {'return_all':True,'maxfev':2000,'maxiter':2000})
    
    # save the obtained parameters
    results[SubID] = mle_model
    len_data_per_sub[SubID] = min_len*len(SNRs)
    
    # plot the evolution of the estimated parameters
    convergence = 'SUCCESSFULL'
    if not mle_model['success']:
        convergence = 'FAILURE'
    vecs = np.array(mle_model['allvecs'])
    plt.figure()
    plt.title('Subject'+realSubIDs[SubID]+' ('+convergence+')')
    for i in range(7):
        plt.plot(vecs[:,i],label=parameters_names[i])
    plt.legend()
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')
    #plt.savefig('Parameters_Convergence_Subject'+realSubIDs[SubID]+'_Bayesian00')
    os.chdir(datapath)
    plt.close()
    
    print('\n... convergence for Subject'+realSubIDs[SubID]+' : ', convergence, '\n result : \n')
    for i in range(len(parameters_names)):
        if parameters_names[i] == 'mu_noise':
            print(parameters_names[i] + ' = ',mle_model.x[i],'\n(fit on noise = ',mu_noise_guess)
        elif parameters_names[i] == 'std_noise':
            print(parameters_names[i] + ' = ',mle_model.x[i],'\n(fit on noise = ',std_noise_guess)
        else :
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
        
    # save model as a DataFrame
    for key in model.keys():
        model[key] = [model[key]]
    model_to_save = pandas.DataFrame(model)
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')
    #model_to_save.to_csv('ModelFit_Subject'+realSubIDs[SubID]+'_Bayesian00')
    os.chdir(datapath)

#%% Plot empirical and fit distributions superposed to evaluate the fit

for SubID in range(20) : 
    
    print('\n ... distribution plot for Subject'+realSubIDs[SubID]+' ...\n')
    
    # get model fit arguments
    m,s = noise_parameters[SubID]
    k, threshsnr, step, L_high, k_high, mu_noise, std_noise = results[SubID]['x']
    convergence = 'SUCCESSFULL'
    if not results[SubID]['success']:
        convergence = 'FAILURE'
    
    # get fit lists for high and low states
    snrs = np.linspace(-20.5,-4.5,100)
    low_values = [m for i in range(100)]
    high_values = []
    for snr in snrs :
        mu_snr = L_high/(1+np.exp(-k_high*(snr-threshsnr))) + step
        high_values.append(mu_snr)
        
    
    # plot
    plt.figure()
    plt.title('Empirical and fit distributions for SubJect'+realSubIDs[SubID]+' ('+convergence+')')
    real_means = []
    for key in empirical_distributions_500ms[SubID].keys():
        real_means.append(np.mean(empirical_distributions_500ms[SubID][key]))
        plt.scatter([key for x in empirical_distributions_500ms[SubID][key]],
                 [x for x in empirical_distributions_500ms[SubID][key]],color='blue',marker='.')
    plt.plot(SNRs,real_means,color='blue',label='empirical means')
    plt.plot(snrs,low_values,color='red',label='low state')
    plt.plot(snrs,high_values,color='green',label='high state')
    plt.legend()
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')
    #plt.savefig('Empirical_VS_Fit_Subject'+realSubIDs[SubID]+'_Bayesian00')
    os.chdir(datapath)
    plt.close()
