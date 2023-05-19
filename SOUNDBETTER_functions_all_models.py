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
import scipy.io
import pandas
from scipy.signal import butter,filtfilt
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

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
SimulateEmpiricalDistribution = seedSOUNDBETTER.SimulateEmpiricalDistribution



#%% _Bifucation00

def fit_Bifucation00(data,iparameters,priors,maxfev=2000,maxiter=2000):
    
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
            
            if snr == -20:
                beta = 0
                mu_snr = mu_noise
            else :
                mu_snr = L_high/(1+np.exp(-k_high*(snr-threshsnr))) + step
                beta = 1/(1+np.exp(-k*(snr-threshsnr)))
                
            for ind, activity in enumerate(data[snr]):
                p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
                p_low = np.exp(-(activity-mu_noise)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
                # likelihood
                p_likelihood = beta*p_high + (1-beta)*p_low
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


def test_Bifucation00(data,parameters):
    
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
            beta = 1/(1+np.exp(-k*(snr-threshsnr)))
            if snr == -20:
                beta = 0
                mu_snr = mu_noise
            p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            p_low = np.exp(-(activity-mu_noise)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            # likelihood
            p_likelihood = (beta*p_high + (1-beta)*p_low)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)
    

def CV5_Bifurcation00(list_SubIDs, save=False, redo=False, nb_optimizations=1):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_k','best_x0','best_step','best_L_high',
                        'best_k_high','best_mu_low','best_sigma']
    
    # initiate output
    Bifurcation00_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if nb_optimizations == 1:
            name_file = 'Model7_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat'
        else : 
            name_file = 'Model7_'+str(nb_optimizations)+'i_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat'
        if os.path.exists(name_file)==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        Bifurcation00_results_this_sub = {para:[] for para in parameters_names}
        Bifurcation00_results_this_sub['time'] = list(range(-500,2000,2))
        Bifurcation00_results_this_sub['lowpassc'] = 10
        Bifurcation00_results_this_sub['TWOI'] = TWOI[:-1]
        Bifurcation00_results_this_sub['exitflags'] = []
        Bifurcation00_results_this_sub['trainLLH'] = []
        Bifurcation00_results_this_sub['testLLH'] = []
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Bifurcation00_results_this_sub_this_time = {para:[] for para in parameters_names}   
            Bifurcation00_results_this_sub_this_time['exitflags'] = []
            Bifurcation00_results_this_sub_this_time['trainLLH'] = []
            Bifurcation00_results_this_sub_this_time['testLLH'] = []
            
            for iblocks in range(5) :
            
                blocks_train = list_of_blocks_train[iblocks]
                blocks_test = list_of_blocks_test[iblocks]
            
                data_train = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
            
                # compute initial parameters
                mu_noise_guess, std_noise_guess = np.mean(data_train[-20]), np.std(data_train[-20])
                L_high_guess = np.mean(data_train[-5]) - np.mean(data_train[-13])
                k_guess, k_high_guess = L_high_guess/8, L_high_guess/8
                step_guess = np.mean(data_train[-13]) - np.mean(data_train[-20])
                threshsnr_guess = -9
                
                iparameters = np.array([k_guess, threshsnr_guess, step_guess, L_high_guess, 
                                        k_high_guess, mu_noise_guess, std_noise_guess])
                priors = [None, None, None, None, None, None, None,]
                
                
                # fit the model with nb_optimizations trials
                success, parameters, fun, nb_points = fit_Bifucation00(data_train, iparameters, priors, 
                                                        maxfev=200*len(iparameters), maxiter=200*len(iparameters))
                success_list = [success]
                for opti in range(nb_optimizations-1):
                    
                    tempt_iparameters = [np.random.normal(ipar,1) for ipar in iparameters]
                    
                    tempt_success, tempt_parameters, tempt_fun, tempt_nb_points = fit_Bifucation00(data_train, tempt_iparameters, priors, 
                                                            maxfev=200*len(iparameters), maxiter=200*len(iparameters))
                    success_list.append(tempt_success)
                    
                    if success == False and tempt_success == True :
                        success, parameters, fun, nb_points = tempt_success, tempt_parameters, tempt_fun, tempt_nb_points
                    if (success == False and tempt_success == False) or (success == True and tempt_success == True):
                        if fun > tempt_fun :
                            success, parameters, fun, nb_points = tempt_success, tempt_parameters, tempt_fun, tempt_nb_points
                    
                
                print(success_list, parameters)
                print('\n log-likelihood on training set : ', fun) # np.exp(-fun/nb_points))
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                fun_test, avg_lh = test_Bifucation00(data_test,parameters)
                
                print('\n log-likelihood for testing set : ', fun_test, '\n' )
                
                # fill-in the output for this time window
                for ind_para, para in enumerate(parameters_names) :
                    Bifurcation00_results_this_sub_this_time[para].append(parameters[ind_para])
                Bifurcation00_results_this_sub_this_time['exitflags'].append(int(success))
                Bifurcation00_results_this_sub_this_time['trainLLH'].append(fun)
                Bifurcation00_results_this_sub_this_time['testLLH'].append(fun_test)
                
            # fill-in the output for this subject
            for key in Bifurcation00_results_this_sub_this_time.keys() :
                Bifurcation00_results_this_sub[key].append(Bifurcation00_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Bifurcation00_results_this_sub['Bifurcation00_LLH'] = np.mean(Bifurcation00_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Bifurcation00_results_this_sub.keys():
                matfile[key] = np.array(Bifurcation00_results_this_sub[key])
            scipy.io.savemat(name_file, matfile)
            
            os.chdir(datapath)
        
        Bifurcation00_results.append(Bifurcation00_results_this_sub)
        
    return(Bifurcation00_results)


#%% _Bifucation01

def fit_Bifucation01(data,iparameters,priors,maxfev=2000,maxiter=2000):
    
    '''
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    iparameters : len 5 list or array of the initial guesses 
        (['k','threshsnr','step','L_high','k_high']).
    priors : len 5 list or array of (mu_param, std_param) tuples or None (if no prior to apply).
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
        k, threshsnr, step, L_high, k_high = parameters
        
        # compute the log_likelihood
        neg_LL = 0
        for snr in SNRs :
            
            if snr == -20:
                beta = 0
                mu_snr = mu_noise_guess
            else :
                mu_snr = L_high/(1+np.exp(-k_high*(snr-threshsnr))) + step
                beta = 1/(1+np.exp(-k*(snr-threshsnr)))
                
            for ind, activity in enumerate(data[snr]):
                p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise_guess**2))/(std_noise_guess*np.sqrt(2*np.pi))
                p_low = np.exp(-(activity-mu_noise_guess)**2/(2*std_noise_guess**2))/(std_noise_guess*np.sqrt(2*np.pi))
                # likelihood
                p_likelihood = beta*p_high + (1-beta)*p_low
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
                         np.array([k_guess, threshsnr_guess, step_guess, L_high_guess, k_high_guess]), 
                         method='Nelder-Mead',
                         options = {'return_all':False,'maxfev':maxfev,'maxiter':maxiter})
    
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    return(mle_model['success'], [x for x in mle_model['x']]+[mu_noise_guess, std_noise_guess], mle_model['fun'], nb_points)


def test_Bifucation01(data,parameters):
    
    '''
    Exactly the same as test_Bifurcation00.
    
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
            beta = 1/(1+np.exp(-k*(snr-threshsnr)))
            if snr == -20:
                beta = 0
                mu_snr = mu_noise
            p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            p_low = np.exp(-(activity-mu_noise)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            # likelihood
            p_likelihood = (beta*p_high + (1-beta)*p_low)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)


def CV5_Bifurcation01(list_SubIDs, priors = [None, None, None, None, None], save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    priors : LIST of (float,float) ie (mean_para,std_para) : priors for the 5 parameters.
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation01_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_k','best_x0','best_step','best_L_high',
                        'best_k_high','best_mu_low','best_sigma']
    
    if priors == [None, None, None, None, None] :
        model_name = '7B'
    else :
        model_name = '7C'
    
    # initiate output
    Bifurcation01_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model'+model_name+'_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        Bifurcation01_results_this_sub = {para:[] for para in parameters_names}
        Bifurcation01_results_this_sub['time'] = list(range(-500,2000,2))
        Bifurcation01_results_this_sub['lowpassc'] = 10
        Bifurcation01_results_this_sub['TWOI'] = TWOI[:-1]
        Bifurcation01_results_this_sub['exitflags'] = []
        Bifurcation01_results_this_sub['trainLLH'] = []
        Bifurcation01_results_this_sub['testLLH'] = []
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Bifurcation01_results_this_sub_this_time = {para:[] for para in parameters_names}   
            Bifurcation01_results_this_sub_this_time['exitflags'] = []
            Bifurcation01_results_this_sub_this_time['trainLLH'] = []
            Bifurcation01_results_this_sub_this_time['testLLH'] = []
            
            for iblocks in range(5) :
            
                blocks_train = list_of_blocks_train[iblocks]
                blocks_test = list_of_blocks_test[iblocks]
            
                data_train = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
            
                # compute initial parameters
                mu_noise_guess, std_noise_guess = np.mean(data_train[-20]), np.std(data_train[-20])
                L_high_guess = np.mean(data_train[-5]) - np.mean(data_train[-13])
                k_guess, k_high_guess = L_high_guess/8, L_high_guess/8
                step_guess = np.mean(data_train[-13]) - np.mean(data_train[-20])
                threshsnr_guess = -9
                
                iparameters = np.array([k_guess, threshsnr_guess, step_guess, L_high_guess, 
                                        k_high_guess, mu_noise_guess, std_noise_guess])
                priors = priors + [None, None,]
                
                success, parameters, fun, nb_points = fit_Bifucation01(data_train, iparameters, priors, 
                                                            maxfev=200*len(iparameters), maxiter=200*len(iparameters))
                
                print(success, parameters)
                print('\n log-likelihood on training set : ', fun) # np.exp(-fun/nb_points))
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                fun_test, avg_lh = test_Bifucation01(data_test,parameters)
                
                print('\n log-likelihood for testing set : ', fun_test, '\n' )
                
                # fill-in the output for this time window
                for ind_para, para in enumerate(parameters_names) :
                    Bifurcation01_results_this_sub_this_time[para].append(parameters[ind_para])
                Bifurcation01_results_this_sub_this_time['exitflags'].append(int(success))
                Bifurcation01_results_this_sub_this_time['trainLLH'].append(fun)
                Bifurcation01_results_this_sub_this_time['testLLH'].append(fun_test)
                
            # fill-in the output for this subject
            for key in Bifurcation01_results_this_sub_this_time.keys() :
                Bifurcation01_results_this_sub[key].append(Bifurcation01_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Bifurcation01_results_this_sub['Bifurcation01_LLH'] = np.mean(Bifurcation01_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Bifurcation01_results_this_sub.keys():
                matfile[key] = np.array(Bifurcation01_results_this_sub[key])
            scipy.io.savemat('Model'+model_name+'_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        Bifurcation01_results.append(Bifurcation01_results_this_sub)
        
    return(Bifurcation01_results)

#%% _Unimodal00

def fit_Unimodal00(data,iparameters,priors,maxfev=2000,maxiter=2000):
    
    '''
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    iparameters : len 6 list or array of the initial guesses 
        (['k','threshsnr', 'mu_maxsnr', 'L', 'slope_sigma', 'intercept_sigma']).
    priors : len 6 list or array of (mu_param, std_param) tuples or None (if no prior to apply).
    maxfev, maxiter : int, optionnal. Max number of function evaluation / iterations. Default 2000.

    Returns
    -------
    success : bool informing about the convergence of the optimization.
    parameters : dict of the final parameters obtained.
    fun : float, final value of p(this_model|data) (or p(data|this_model) if no prior).
    
    '''
    
    # unpack the initial parameters
    k_guess,threshsnr_guess, mu_maxsnr_guess, L_guess, slope_sigma_guess, intercept_sigma_guess = iparameters
    
    # pre - function to minimize
    def pre_MLE_unimodal(data,parameters):
        # extract parameters
        k, threshsnr, mu_maxsnr, L, slope_sigma, intercept_sigma = parameters
        
        # compute the log_likelihood
        neg_LL = 0
        for snr in SNRs :
            
            mu_snr = L/(1+np.exp(-k*(snr-threshsnr))) - L/(1+np.exp(-k*(-5-threshsnr))) + mu_maxsnr
            sigma_snr = slope_sigma*mu_snr + intercept_sigma
                
            for ind, activity in enumerate(data[snr]):
                # likelihood
                p_likelihood = scipy.stats.norm.pdf(activity,mu_snr,sigma_snr)
                neg_LL -= np.log(p_likelihood)
                # prior probabilities
                for iparam, prior in enumerate(priors) :
                    if prior != None :
                        param = parameters[iparam]
                        mu_param, std_param = prior
                        prior = scipy.stats.norm.pdf(param,mu_param,std_param)
                        neg_LL -= np.log(prior) 
        return(neg_LL)
    
    # function to minimize
    def MLE_unimodal(parameters):
        neg_LL = pre_MLE_unimodal(data,parameters)
        return(neg_LL)
    
    # minimize fun in the parameters space
    mle_model = minimize(MLE_unimodal, 
                         np.array([k_guess,threshsnr_guess, mu_maxsnr_guess, L_guess, slope_sigma_guess, intercept_sigma_guess]), 
                         method='Nelder-Mead',
                         options = {'return_all':False,'maxfev':maxfev,'maxiter':maxiter})
    
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    return(mle_model['success'], mle_model['x'], mle_model['fun'], nb_points)


def test_Unimodal00(data,parameters):
    
    '''
    
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 6 list or array of already fitted parameters.

    Returns
    -------
    neg_LL : float
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    k, threshsnr, mu_maxsnr, L, slope_sigma, intercept_sigma = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for snr in SNRs :
        
        mu_snr = L/(1+np.exp(-k*(snr-threshsnr))) - L/(1+np.exp(-k*(-5-threshsnr))) + mu_maxsnr
        sigma_snr = slope_sigma*mu_snr + intercept_sigma
            
        for ind, activity in enumerate(data[snr]):
            # likelihood
            p_likelihood = scipy.stats.norm.pdf(activity,mu_snr,sigma_snr)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)


def test_noNoise_Unimodal00(data,parameters):
    
    '''
    
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 6 list or array of already fitted parameters.

    Returns
    -------
    neg_LL : float
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    k, threshsnr, mu_maxsnr, L, slope_sigma, intercept_sigma = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for snr in SNRs[1:] :
        
        mu_snr = L/(1+np.exp(-k*(snr-threshsnr))) - L/(1+np.exp(-k*(-5-threshsnr))) + mu_maxsnr
        sigma_snr = slope_sigma*mu_snr + intercept_sigma
            
        for ind, activity in enumerate(data[snr]):
            # likelihood
            p_likelihood = scipy.stats.norm.pdf(activity,mu_snr,sigma_snr)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)



def CV5_Unimodal00(list_SubIDs, save=False, redo=False, nb_optimizations=1):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_k', 'best_threshsnr', 'best_mu_maxsnr', 'best_L', 'best_slope_sigma', 'best_intercept_sigma']
    
    # initiate output
    Unimodal_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if nb_optimizations == 1:
            name_file = 'Model2B_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat'
        else :
            name_file = 'Model2B_'+str(nb_optimizations)+'i_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat'
        if os.path.exists(name_file)==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        Unimodal00_results_this_sub = {para:[] for para in parameters_names}
        Unimodal00_results_this_sub['time'] = list(range(-500,2000,2))
        Unimodal00_results_this_sub['lowpassc'] = 10
        Unimodal00_results_this_sub['TWOI'] = TWOI[:-1]
        Unimodal00_results_this_sub['exitflags'] = []
        Unimodal00_results_this_sub['trainLLH'] = []
        Unimodal00_results_this_sub['testLLH'] = []
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Unimodal00_results_this_sub_this_time = {para:[] for para in parameters_names}   
            Unimodal00_results_this_sub_this_time['exitflags'] = []
            Unimodal00_results_this_sub_this_time['trainLLH'] = []
            Unimodal00_results_this_sub_this_time['testLLH'] = []
            
            for iblocks in range(5) :
            
                blocks_train = list_of_blocks_train[iblocks]
                blocks_test = list_of_blocks_test[iblocks]
            
                data_train = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
            
                # compute initial parameters
                slope_sigma_guess = (np.std(data_train[-5]) - np.std(data_train[-13]))/8
                # approx. of sigma(snr) : slope_sigma*(snr-(-13)) + sigma(-13)
                intercept_sigma_guess = 13*slope_sigma_guess + np.std(data_train[-13]) # approx_sigma(0dB)
                k_guess = (np.mean(data_train[-5]) - np.mean(data_train[-13]))/8
                threshsnr_guess = -9
                mu_maxsnr_guess = np.mean(data_train[-5])
                L_guess = 2*mu_maxsnr_guess
                
                iparameters = np.array([k_guess, threshsnr_guess, mu_maxsnr_guess, L_guess, slope_sigma_guess, intercept_sigma_guess])
                priors = [None, None, None, None, None, None]
                
                # fit the model with nb_optimizations trials
                success, parameters, fun, nb_points = fit_Unimodal00(data_train, iparameters, priors, 
                                                        maxfev=200*len(iparameters), maxiter=200*len(iparameters))
                success_list = [success] # just to plot in real_time, not used for the script
                for opti in range(nb_optimizations-1):
                    
                    tempt_iparameters = [np.random.normal(ipar,1) for ipar in iparameters]
                    
                    tempt_success, tempt_parameters, tempt_fun, tempt_nb_points = fit_Unimodal00(data_train, tempt_iparameters, priors, 
                                                            maxfev=200*len(iparameters), maxiter=200*len(iparameters))
                    success_list.append(tempt_success)
                    
                    if success == False and tempt_success == True :
                        success, parameters, fun, nb_points = tempt_success, tempt_parameters, tempt_fun, tempt_nb_points
                    if (success == False and tempt_success == False) or (success == True and tempt_success == True):
                        if fun > tempt_fun :
                            success, parameters, fun, nb_points = tempt_success, tempt_parameters, tempt_fun, tempt_nb_points
                
                print(success_list, parameters)
                print('\n log-likelihood on training set : ', fun) # np.exp(-fun/nb_points))
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                fun_test, avg_lh = test_Unimodal00(data_test,parameters)
                
                print('\n log-likelihood for testing set : ', fun_test, '\n' )
                
                # fill-in the output for this time window
                for ind_para, para in enumerate(parameters_names) :
                    Unimodal00_results_this_sub_this_time[para].append(parameters[ind_para])
                Unimodal00_results_this_sub_this_time['exitflags'].append(int(success))
                Unimodal00_results_this_sub_this_time['trainLLH'].append(fun)
                Unimodal00_results_this_sub_this_time['testLLH'].append(fun_test)
                
            # fill-in the output for this subject
            for key in Unimodal00_results_this_sub_this_time.keys() :
                Unimodal00_results_this_sub[key].append(Unimodal00_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Unimodal00_results_this_sub['Bifurcation00_LLH'] = np.mean(Unimodal00_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Unimodal00_results_this_sub.keys():
                matfile[key] = np.array(Unimodal00_results_this_sub[key])
            scipy.io.savemat('Model2B_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        Unimodal_results.append(Unimodal00_results_this_sub)
        
    return(Unimodal_results)


def rewrite_CV5_noNoise_Unimodal00(list_SubIDs, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_k', 'best_threshsnr', 'best_mu_maxsnr', 'best_L', 'best_slope_sigma', 'best_intercept_sigma']
    
    # initiate output
    Unimodal_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model2B_noNoise_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        # load the result of CV5_Unimodal00
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        current_results = scipy.io.loadmat('Model2B_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')
        
        
        Unimodal00_results_this_sub = {para: current_results[para] for para in parameters_names}
        Unimodal00_results_this_sub['testLLH'] = [] # only this one really matters here
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Unimodal00_results_this_sub_this_time = {'testLLH': []}
            
            for iblocks in range(5) :
            
                blocks_test = list_of_blocks_test[iblocks]
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                parameters = [Unimodal00_results_this_sub[para][ind_time][iblocks] for para in parameters_names]
                fun_test, avg_lh = test_noNoise_Unimodal00(data_test,parameters)
                
                print('\n log-likelihood for testing set : ', fun_test, '\n' )
                
                # fill-in the output for this time window
                Unimodal00_results_this_sub_this_time['testLLH'].append(fun_test)
                
            # fill-in the output for this subject
            for key in Unimodal00_results_this_sub_this_time.keys() :
                Unimodal00_results_this_sub[key].append(Unimodal00_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Unimodal00_results_this_sub['Bifurcation00_LLH'] = np.mean(Unimodal00_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Unimodal00_results_this_sub.keys():
                matfile[key] = np.array(Unimodal00_results_this_sub[key])
            scipy.io.savemat('Model2B_noNoise_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        Unimodal_results.append(Unimodal00_results_this_sub)
        
    return(Unimodal_results)


#%% _Null00


def fit_Null00(data,iparameters,priors,maxfev=2000,maxiter=2000):
    
    '''
    
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    iparameters : len 1 list or array of the initial guesses 
        (['sigma']).
    priors : len 1 list or array of (mu_param, std_param) tuples or None (if no prior to apply).
    maxfev, maxiter : int, optionnal. Max number of function evaluation / iterations. Default 2000.

    Returns
    -------
    success : bool informing about the convergence of the optimization.
    parameters : dict of the final parameters obtained.
    fun : float, final value of p(this_model|data) (or p(data|this_model) if no prior).
    
    '''
    
    # analytical solution for mu
    all_activity = []
    for snr in SNRs :
        for x in data[snr] :
            all_activity.append(x)
    mu = np.mean(all_activity)
    
    # unpack the initial parameters
    sigma_guess = iparameters
    
    # pre - function to minimize
    def pre_MLE_null(data,parameters):
        # extract parameters
        sigma = parameters
        
        # compute the log_likelihood
        neg_LL = 0
        for snr in SNRs :
                
            for ind, activity in enumerate(data[snr]):
                # likelihood
                p_likelihood = scipy.stats.norm.pdf(activity,mu,sigma)
                neg_LL -= np.log(p_likelihood)
                # prior probabilities
                for iparam, prior in enumerate(priors) :
                    if prior != None :
                        param = parameters[iparam]
                        mu_param, std_param = prior
                        prior = scipy.stats.norm.pdf(param,mu_param,std_param)
                        neg_LL -= np.log(prior) 
        return(neg_LL)
    
    # function to minimize
    def MLE_null(parameters):
        neg_LL = pre_MLE_null(data,parameters)
        return(neg_LL)
    
    # minimize fun in the parameters space
    mle_model = minimize(MLE_null, 
                         np.array([sigma_guess]), 
                         method='Nelder-Mead',
                         options = {'return_all':False,'maxfev':maxfev,'maxiter':maxiter})
    
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    return(mle_model['success'], [mu, mle_model['x'][0]], mle_model['fun'], nb_points)


def test_Null00(data,parameters):
    
    '''
    
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 1 list or array of already fitted parameters.

    Returns
    -------
    neg_LL : float
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    mu, sigma = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for snr in SNRs :
            
        for ind, activity in enumerate(data[snr]):
            # likelihood
            p_likelihood = scipy.stats.norm.pdf(activity,mu,sigma)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)


def test_noNoise_Null00(data,parameters):
    
    '''
    
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 1 list or array of already fitted parameters.

    Returns
    -------
    neg_LL : float
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    mu, sigma = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for snr in SNRs[1:] :
            
        for ind, activity in enumerate(data[snr]):
            # likelihood
            p_likelihood = scipy.stats.norm.pdf(activity,mu,sigma)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)


def CV5_Null00(list_SubIDs, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_mu0','best_sigma0']
    
    # initiate output
    Null_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model0_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        Null00_results_this_sub = {para:[] for para in parameters_names}
        Null00_results_this_sub['time'] = list(range(-500,2000,2))
        Null00_results_this_sub['lowpassc'] = 10
        Null00_results_this_sub['TWOI'] = TWOI[:-1]
        Null00_results_this_sub['exitflags'] = []
        Null00_results_this_sub['trainLLH'] = []
        Null00_results_this_sub['testLLH'] = []
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Null00_results_this_sub_this_time = {para:[] for para in parameters_names}   
            Null00_results_this_sub_this_time['exitflags'] = []
            Null00_results_this_sub_this_time['trainLLH'] = []
            Null00_results_this_sub_this_time['testLLH'] = []
            
            for iblocks in range(5) :
            
                blocks_train = list_of_blocks_train[iblocks]
                blocks_test = list_of_blocks_test[iblocks]
            
                data_train = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
            
                # compute initial parameters
                sigma_guess = 1
                
                iparameters = np.array([sigma_guess])
                priors = [None]
                
                success, parameters, fun, nb_points = fit_Null00(data_train, iparameters, priors, 
                                                            maxfev=200*len(iparameters), maxiter=200*len(iparameters))
                
                print(success, parameters)
                print('\n log-likelihood on training set : ', fun) # np.exp(-fun/nb_points))
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                fun_test, avg_lh = test_Null00(data_test,parameters)
                
                print('\n log-likelihood for testing set : ', fun_test, '\n' )
                
                # fill-in the output for this time window
                for ind_para, para in enumerate(parameters_names) :
                    Null00_results_this_sub_this_time[para].append(parameters[ind_para])
                Null00_results_this_sub_this_time['exitflags'].append(int(success))
                Null00_results_this_sub_this_time['trainLLH'].append(fun)
                Null00_results_this_sub_this_time['testLLH'].append(fun_test)
                
            # fill-in the output for this subject
            for key in Null00_results_this_sub_this_time.keys() :
                Null00_results_this_sub[key].append(Null00_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Null00_results_this_sub['Null00_LLH'] = np.mean(Null00_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Null00_results_this_sub.keys():
                matfile[key] = np.array(Null00_results_this_sub[key])
            scipy.io.savemat('Model0_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        Null_results.append(Null00_results_this_sub)
        
    return(Null_results)


def rewrite_CV5_noNoise_Null00(list_SubIDs, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Null00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_mu0','best_sigma0']
    
    # initiate output
    Null_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model0_noNoise_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        # load the result of CV5_Null00
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        current_results = scipy.io.loadmat('Model0_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')
        
        Null00_results_this_sub = {para: current_results[para] for para in parameters_names}
        Null00_results_this_sub['testLLH'] = []
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Null00_results_this_sub_this_time = {'testLLH' : []}
            
            for iblocks in range(5) :
            
                blocks_test = list_of_blocks_test[iblocks]
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                parameters = [Null00_results_this_sub[para][ind_time][iblocks] for para in parameters_names]
                fun_test, avg_lh = test_noNoise_Null00(data_test,parameters)
                
                print('\n log-likelihood for testing set : ', fun_test, '\n' )
                
                # fill-in the output for this time window
                Null00_results_this_sub_this_time['testLLH'].append(fun_test)
                
            # fill-in the output for this subject
            for key in Null00_results_this_sub_this_time.keys() :
                Null00_results_this_sub[key].append(Null00_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Null00_results_this_sub['Null00_LLH'] = np.mean(Null00_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Null00_results_this_sub.keys():
                matfile[key] = np.array(Null00_results_this_sub[key])
            scipy.io.savemat('Model0_noNoise_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        Null_results.append(Null00_results_this_sub)
        
    return(Null_results)

#%% _PolyBayesian00

def fit_PolyBayesian00(data,iparameters,priors,maxfev=2000,maxiter=2000):
    
    '''
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    iparameters : len (deg + 5) list or array of the initial guesses
        (! COEF IN THE ORDER OF GROWING DEGREE ! (ex : c,b,a))
        ([.,'threshsnr','k','mu_noise','std_noise']).
    priors : len (deg + 5) list or array of (mu_param, std_param) tuples or None (if no prior to apply).
    deg : int, degree of the polynome to fit.
    maxfev, maxiter : int, optionnal. Max number of function evaluation / iterations. Default 2000.

    Returns
    -------
    success : bool informing about the convergence of the optimization.
    parameters : dict of the final parameters obtained.
    fun : float, final value of p(this_model|data) (or p(data|this_model) if no prior).
    
    '''
    
    # unpack the initial parameters
    if len(iparameters) != len(priors):
        raise ValueError('len(priors) != len(iparameters)')
    
    # pre - function to minimize
    def pre_MLE_bifurcation(data,parameters):
        
        # extract parameters
        poly_coefs, (threshsnr,k,mu_noise,std_noise) = parameters[:-4], parameters[-4:]
        deg = len(poly_coefs) - 1
        
        # compute the log_likelihood
        neg_LL = 0
        for snr in SNRs :
            for ind, activity in enumerate(data[snr]):
                mu_snr = np.sum([poly_coefs[j]*activity**j for j in range(deg+1)])
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
                         iparameters,
                         method='Nelder-Mead',
                         options = {'return_all':False,'maxfev':maxfev,'maxiter':maxiter})
    
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    return(mle_model['success'], mle_model['x'], mle_model['fun'], nb_points)


def test_PolyBayesian00(data,parameters):
    
    '''
     
    Parameters
    ----------
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len (deg + 5) list or array of already fitted parameters.

    Returns
    -------
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    poly_coefs, (threshsnr,k,mu_noise,std_noise) = parameters[:-4], parameters[-4:]
    deg = len(poly_coefs) - 1
    
    # compute the log_likelihood
    neg_LL = 0
    for snr in SNRs :
        for ind, activity in enumerate(data[snr]):
            mu_snr = np.sum([poly_coefs[j]*activity**j for j in range(deg+1)])
            p_high = np.exp(-(activity-mu_snr)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            p_low = np.exp(-(activity-mu_noise)**2/(2*std_noise**2))/(std_noise*np.sqrt(2*np.pi))
            beta = 1/(1+np.exp(-k*(snr-threshsnr)))
            # likelihood
            p_likelihood = (beta*p_high + (1-beta)*p_low)
            neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = np.sum([len(data[k]) for k in data.keys()])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)
    

#%% _Bayesian00


def fit_Bayesian00(snr,data,iparameters,priors,maxiter=2000):
    
    '''
    
    Uses the EM method instead of MLE 
    (https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture).   
     
    Parameters
    ----------
    snr : negative int, the one snr on which to do the analysis
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    iparameters : len 5 list or array of the initial guesses 
        (['beta','mu_low','sigma_low','mu_high','sigma_high']).
    priors : len 5 list or array of (mu_param, std_param) tuples or None (if no prior to apply).
    maxfev, maxiter : int, optionnal. Max number of function evaluation / iterations. Default 2000.

    Returns
    -------
    success : bool informing about the convergence of the optimization.
    HSP_list (high-state probability) : list with proba of being in high state for each trial.
    parameters : dict of the final parameters obtained.
    fun : float, final value of p(this_model|data) (or p(data|this_model) if no prior).
    
    '''
    
    # unpack the initial parameters
    beta_guess, mu_low_guess, sigma_low_guess, mu_high_guess, sigma_high_guess = iparameters
    beta, mu_low, sigma_low, mu_high, sigma_high = [beta_guess], [mu_low_guess], [sigma_low_guess], [mu_high_guess], [sigma_high_guess]
    
    # compute probabilities of belonging to the higher state (HSP, High-State Probability) based on guessed parameters
    HSP_list = []
    for x in data[snr] :
        a, b = norm.pdf(x,mu_high[-1],sigma_high[-1]), norm.pdf(x,mu_low[-1],sigma_low[-1])
        HSP_list.append(beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
    
    # iterate over the parameters' versions
    LLs = []
    convergence = 0
    i = 0
    while i<maxiter and convergence < 10 :
            
        # update the parameters
        beta.append(np.mean(HSP_list))
        mu_high.append(np.sum([HSP_list[k]*data[snr][k] for k in range(len(data[snr]))])/np.sum(HSP_list))
        sigma_high.append(np.sum([HSP_list[k]*(data[snr][k]-mu_high[-1])**2 for k in range(len(data[snr]))])/np.sum(HSP_list))
        mu_low.append(np.sum([(1-HSP_list[k])*data[snr][k] for k in range(len(data[snr]))])/np.sum([1-hsp for hsp in HSP_list]))
        sigma_low.append(np.sum([(1-HSP_list[k])*(data[snr][k]-mu_low[-1])**2 for k in range(len(data[snr]))])/np.sum([1-hsp for hsp in HSP_list]))
        
        # update HSP_list      
        for ind_x, x in enumerate(data[snr]) :
            a, b = norm.pdf(x,mu_high[-1],sigma_high[-1]), norm.pdf(x,mu_low[-1],sigma_low[-1])
            HSP_list[ind_x] = (beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
            
        # compute the obtained log-likelihood
        new_LL = 0
        for ind_x, x in enumerate(data[snr]) :
            if beta[-1] >= 0.5 :
                new_LL -= np.log(beta[-1]) - 0.5*np.log(sigma_high[-1]) - 0.5*((x-mu_high[-1])**2)/sigma_high[-1] - 0.5*np.log(2*np.pi)
            else :
                new_LL -= np.log(1-beta[-1]) - 0.5*np.log(sigma_low[-1]) - 0.5*((x-mu_low[-1])**2)/sigma_low[-1] - 0.5*np.log(2*np.pi)
        LLs.append(new_LL)
        
        # assess convergence
        if len(LLs) > 1 :
            if LLs[-2] - LLs[-1] < 1:
                convergence += 1
            else :
                convergence = 0
            
        i += 1
    
    
    nb_points = len(data[snr])
    
    return(convergence>=10, HSP_list, [beta[-1], mu_low[-1], sigma_low[-1], mu_high[-1], sigma_high[-1]], LLs[-1], nb_points)



def test_Bayesian00(snr,data,parameters):
    
    '''
    
    Compute the LL considering the final prediction of the EM algorithm to be true.
    
    Parameters
    ----------
    snr : negative int, the one snr on which to do the analysis
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 5 list or array of already fitted parameters.

    Returns
    -------
    neg_LL : float
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    beta, mu_low, sigma_low, mu_high, sigma_high = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for ind, activity in enumerate(data[snr]):
        # likelihood
        a, b = norm.pdf(activity,mu_high,sigma_high), norm.pdf(activity,mu_low,sigma_low)
        HSP = beta*a/(beta*a + (1-beta)*b)
        if HSP >= 0.5:
            p_likelihood = scipy.stats.norm.pdf(activity,mu_high,sigma_high)
        else :
            p_likelihood = scipy.stats.norm.pdf(activity,mu_low,sigma_low)
        neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = len(data[snr])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)


def CV5_Bayesian00(list_SubIDs, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_beta','best_mu_low','best_sigma_low','best_mu_high','best_sigma_high']
    
    betas_per_snr = {-20 : 0, -13 : 0.1, -11 : 0.25, -9 : 0.4, -7 : 0.6, -5 : 0.9}
    
    # initiate output
    Bayesian_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model8_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        Bayesian00_results_this_sub = {para:[] for para in parameters_names}
        Bayesian00_results_this_sub['time'] = list(range(-500,2000,2))
        Bayesian00_results_this_sub['lowpassc'] = 10
        Bayesian00_results_this_sub['TWOI'] = TWOI[:-1]
        Bayesian00_results_this_sub['exitflags'] = []
        Bayesian00_results_this_sub['trainLLH'] = []
        Bayesian00_results_this_sub['testLLH'] = []
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Bayesian00_results_this_sub_this_time = {para:[] for para in parameters_names}   
            Bayesian00_results_this_sub_this_time['exitflags'] = []
            Bayesian00_results_this_sub_this_time['trainLLH'] = []
            Bayesian00_results_this_sub_this_time['testLLH'] = []
            
            for iblocks in range(5) :
            
                blocks_train = list_of_blocks_train[iblocks]
                blocks_test = list_of_blocks_test[iblocks]
            
                data_train = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
            
                success_list, parameters_list, fun_list, nb_points_list = [], [], [], []
            
                for snr in SNRs :
                    
                    # compute initial parameters
                    mu_low_guess, mu_high_guess = np.mean(data_train[-20]), np.mean(data_train[-5]),
                    sigma_low_guess, sigma_high_guess = 1, 1
                    beta_guess = betas_per_snr[snr]
                
                    iparameters = np.array([beta_guess, mu_low_guess, sigma_low_guess, mu_high_guess, sigma_high_guess])
                    priors = [None, None, None, None, None]
                    
                    # train the parameters
                    success, HSP_list, parameters, fun, nb_points = fit_Bayesian00(snr, data_train, iparameters, priors,
                                                                     maxiter=200*len(iparameters))
                    success_list.append(success)
                    parameters_list.append(parameters)
                    fun_list.append(fun)
                    nb_points_list.append(nb_points)
                
                print(success_list, parameters_list)
                print('\n log-likelihood on training set : ', fun_list) # np.exp(-fun/nb_points))
                
                fun_test_list, avg_lh_list = [], []
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                for ind_snr, snr in enumerate(SNRs) :
                    
                    # test the parameters
                    fun_test, avg_lh = test_Bayesian00(snr,data_test,parameters_list[ind_snr])
                    fun_test_list.append(fun_test)
                    avg_lh_list.append(avg_lh)
                
                fun_test = np.sum(fun_test_list)
                print('\n log-likelihood for testing set : ', fun_test_list, '\n' )
                
                # fill-in the output for this time window
                for ind_para, para in enumerate(parameters_names) :
                    Bayesian00_results_this_sub_this_time[para].append(np.array(parameters_list)[:,ind_para])
                Bayesian00_results_this_sub_this_time['exitflags'].append([int(s) for s in success_list])
                Bayesian00_results_this_sub_this_time['trainLLH'].append(fun_list)
                Bayesian00_results_this_sub_this_time['testLLH'].append(np.sum(fun_test_list))
                
            # fill-in the output for this subject
            for key in Bayesian00_results_this_sub_this_time.keys() :
                Bayesian00_results_this_sub[key].append(Bayesian00_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Bayesian00_results_this_sub['Bayesian00_LLH'] = np.mean(Bayesian00_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Bayesian00_results_this_sub.keys():
                matfile[key] = np.array(Bayesian00_results_this_sub[key])
            scipy.io.savemat('Model8_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        Bayesian_results.append(Bayesian00_results_this_sub)
        
    return(Bayesian_results)


#%% _Bayesian01


def fit_Bayesian01(snr,data,iparameters,priors,maxiter=2000):
    
    '''
    YET TO BE TESTED.
    
    Uses the EM method instead of usual MLE 
    (https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture).   
     
    Parameters
    ----------
    snr : negative int, the one snr on which to do the analysis
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    iparameters : len 4 list or array of the initial guesses 
        (['beta','mu_low','mu_high','sigma']).
    priors : len 4 list or array of (mu_param, std_param) tuples or None (if no prior to apply).
    maxfev, maxiter : int, optionnal. Max number of function evaluation / iterations. Default 2000.

    Returns
    -------
    success : bool informing about the convergence of the optimization.
    HSP_list (high-state probability) : list with proba of being in high state for each trial.
    parameters : dict of the final parameters obtained.
    fun : float, final value of p(this_model|data) (or p(data|this_model) if no prior).
    
    '''
    
    # unpack the initial parameters
    beta_guess, mu_low_guess, mu_high_guess, sigma_guess = iparameters
    beta, mu_low, mu_high, sigma = [beta_guess], [mu_low_guess], [mu_high_guess], [sigma_guess]
    
    # compute probabilities of belonging to the higher state based on guessed parameters
    HSP_list = []
    for x in data[snr] :
        a, b = norm.pdf(x,mu_high[-1],sigma[-1]), norm.pdf(x,mu_low[-1],sigma[-1])
        HSP_list.append(beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
    
    # iterate over the parameters' versions
    LLs = []
    convergence = 0
    i = 0
    while i<maxiter and convergence < 10 :
            
        # update the parameters
        beta.append(np.mean(HSP_list))
        mu_high.append(np.sum([HSP_list[k]*data[snr][k] for k in range(len(data[snr]))])/np.sum(HSP_list))
        sigma_high_temporary = np.sum([HSP_list[k]*(data[snr][k]-mu_high[-1])**2 for k in range(len(data[snr]))])/np.sum(HSP_list)
        sigma_low_temporary = np.sum([(1-HSP_list[k])*(data[snr][k]-mu_low[-1])**2 for k in range(len(data[snr]))])/np.sum([1-hsp for hsp in HSP_list])
        sigma.append((sigma_high_temporary + sigma_low_temporary)/2) # average of the supposed stds for the two distributions
        mu_low.append(np.sum([(1-HSP_list[k])*data[snr][k] for k in range(len(data[snr]))])/np.sum([1-hsp for hsp in HSP_list]))
        
        # update HSP_list      
        for ind_x, x in enumerate(data[snr]) :
            a, b = norm.pdf(x,mu_high[-1],sigma[-1]), norm.pdf(x,mu_low[-1],sigma[-1])
            HSP_list[ind_x] = (beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
            
        # compute the obtained log-likelihood
        new_LL = 0
        for ind_x, x in enumerate(data[snr]) :
            if beta[-1] >= 0.5 :
                new_LL -= np.log(beta[-1]) - 0.5*np.log(sigma[-1]) - 0.5*((x-mu_high[-1])**2)/sigma[-1] - 0.5*np.log(2*np.pi)
            else :
                new_LL -= np.log(1-beta[-1]) - 0.5*np.log(sigma[-1]) - 0.5*((x-mu_low[-1])**2)/sigma[-1] - 0.5*np.log(2*np.pi)
        LLs.append(new_LL)
        
        # assess convergence
        if len(LLs) > 1 :
            if LLs[-2] - LLs[-1] < 1:
                convergence += 1
            else :
                convergence = 0
            
        i += 1
    
    
    nb_points = len(data[snr])
    
    return(convergence>=10, HSP_list, [beta[-1], mu_low[-1], mu_high[-1], sigma[-1]], LLs[-1], nb_points)


def test_Bayesian01(snr,data,parameters):
    
    '''
    
    Compute the LL considering the final prediction of the EM algorithm to be true.
    
    Parameters
    ----------
    snr : negative int, the one snr on which to do the analysis
    data : dict obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 4 list or array of already fitted parameters.

    Returns
    -------
    neg_LL : float
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    beta, mu_low, mu_high, sigma = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for ind, activity in enumerate(data[snr]):
        # likelihood
        a, b = norm.pdf(activity,mu_high,sigma), norm.pdf(activity,mu_low,sigma)
        HSP = beta*a/(beta*a + (1-beta)*b)
        if HSP >= 0.5:
            p_likelihood = scipy.stats.norm.pdf(activity,mu_high,sigma)
        else :
            p_likelihood = scipy.stats.norm.pdf(activity,mu_low,sigma)
        neg_LL -= np.log(p_likelihood)
            
    # compute the total number of points
    nb_points = len(data[snr])
    
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)



def CV5_Bayesian01(list_SubIDs, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    TWOI = range(101,901,15)
    
    parameters_names = ['best_beta','best_mu_low','best_mu_high','best_sigma']
    
    betas_per_snr = {-20 : 0, -13 : 0.1, -11 : 0.25, -9 : 0.4, -7 : 0.6, -5 : 0.9}
    
    # initiate output
    Bayesian_results = []
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model8B_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        Bayesian01_results_this_sub = {para:[] for para in parameters_names}
        Bayesian01_results_this_sub['time'] = list(range(-500,2000,2))
        Bayesian01_results_this_sub['lowpassc'] = 10
        Bayesian01_results_this_sub['TWOI'] = TWOI[:-1]
        Bayesian01_results_this_sub['exitflags'] = []
        Bayesian01_results_this_sub['trainLLH'] = []
        Bayesian01_results_this_sub['testLLH'] = []
    
        for ind_time in range(len(TWOI)-1):
            
            print('\n Sub'+realSubIDs[SubID]+', TWOI : '+str(TWOI[ind_time]))
            
            times = (TWOI[ind_time],TWOI[ind_time + 1])
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            Bayesian01_results_this_sub_this_time = {para:[] for para in parameters_names}   
            Bayesian01_results_this_sub_this_time['exitflags'] = []
            Bayesian01_results_this_sub_this_time['trainLLH'] = []
            Bayesian01_results_this_sub_this_time['testLLH'] = []
            
            for iblocks in range(5) :
            
                blocks_train = list_of_blocks_train[iblocks]
                blocks_test = list_of_blocks_test[iblocks]
            
                data_train = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
            
                success_list, parameters_list, fun_list, nb_points_list = [], [], [], []
            
                for snr in SNRs[1:] :
                    
                    # compute initial parameters
                    mu_low_guess, mu_high_guess = np.mean(data_train[-20]), np.mean(data_train[-5]),
                    sigma_guess = 1
                    beta_guess = betas_per_snr[snr]
                
                    iparameters = np.array([beta_guess, mu_low_guess, mu_high_guess, sigma_guess])
                    priors = [None, None, None, None]
                
                    success, HSP_list, parameters, fun, nb_points = fit_Bayesian01(snr, data_train, iparameters, priors,
                                                                     maxiter=200*len(iparameters))
                    success_list.append(success)
                    parameters_list.append(parameters)
                    fun_list.append(fun)
                    nb_points_list.append(nb_points)
                
                print(success_list, parameters_list)
                print('\n log-likelihood on training set : ', fun_list) # np.exp(-fun/nb_points))
                
                fun_test_list, avg_lh_list = [], []
                
                data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                
                for ind_snr, snr in enumerate(SNRs[1:]) :
                    
                    fun_test, avg_lh = test_Bayesian01(snr,data_test,parameters_list[ind_snr])
                    fun_test_list.append(fun_test)
                    avg_lh_list.append(avg_lh)
                
                fun_test = np.sum(fun_test_list)
                print('\n log-likelihood for testing set : ', fun_test_list, '\n' )
                
                # fill-in the output for this time window
                for ind_para, para in enumerate(parameters_names) :
                    Bayesian01_results_this_sub_this_time[para].append(np.array(parameters_list)[:,ind_para])
                Bayesian01_results_this_sub_this_time['exitflags'].append([int(s) for s in success_list])
                Bayesian01_results_this_sub_this_time['trainLLH'].append(fun_list)
                Bayesian01_results_this_sub_this_time['testLLH'].append(np.sum(fun_test_list))
                
            # fill-in the output for this subject
            for key in Bayesian01_results_this_sub_this_time.keys() :
                Bayesian01_results_this_sub[key].append(Bayesian01_results_this_sub_this_time[key])
        
        # average the test_LLH to obtain the model's average LLH
        Bayesian01_results_this_sub['Bayesian01_LLH'] = np.mean(Bayesian01_results_this_sub['testLLH'],1)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            
            matfile = {}
            for key in Bayesian01_results_this_sub.keys():
                matfile[key] = np.array(Bayesian01_results_this_sub[key])
            scipy.io.savemat('Model8B_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        Bayesian_results.append(Bayesian01_results_this_sub)
        
    return(Bayesian_results)

#%% _GaussianMixture00


def fit_GaussianMixture00(data,iparameters,maxiter=2000):
    
    '''
    YET TO BE TESTED.
    
    Uses the EM method instead of MLE 
    (https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture).   
     
    Parameters
    ----------
    data : list of nb_timepoints-dimensional vectors (lists or np.darray ?)
    iparameters : list of [len 5 list or array of the initial guesses 
        (['beta','mu_low','sigma_low','mu_high','sigma_high'])].
    priors : list of [len 5 list or array of (mu_param, std_param) tuples or None (if no prior to apply)].
    maxiter : int, optionnal. Max number of iterations. Default 2000.

    Returns
    -------
    success : list of [bool informing about the convergence of the optimization].
    HSP_list (high-state probability) : list with proba of being in high state for each trial.
    parameters : dict of the final parameters obtained.
    fun : list of [float, final value of p(this_model|data) (or p(data|this_model) if no prior)].
    
    '''
    
    # unpack the initial parameters
    beta_guess, mu_low_guess, sigma_low_guess, mu_high_guess, sigma_high_guess = iparameters
    beta, mu_low, sigma_low, mu_high, sigma_high = [beta_guess], [mu_low_guess], [sigma_low_guess], [mu_high_guess], [sigma_high_guess]
    nb_timepoints = len(mu_low_guess)
    
    # compute probabilities of belonging to the higher state based on guessed parameters
    HSP_list = [] # will contain one value per timepoint
    for ind_data, x in enumerate(data) :
        try :
            a, b = multivariate_normal.pdf(x,mu_high[-1],sigma_high[-1]), multivariate_normal.pdf(x,mu_low[-1],sigma_low[-1])
        except : # if singular matrix
            a, b = multivariate_normal.pdf(x,mu_high[-1],sigma_high[-1],allow_singular=True), multivariate_normal.pdf(x,mu_low[-1],sigma_low[-1],allow_singular=True)
        HSP_list.append(beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
    
    # iterate over the parameters' versions
    LLs = []
    convergence = 0
    i = 0
    while i<maxiter and convergence<20:
        
        # update the parameters
        beta.append(np.mean(HSP_list))
        mu_high.append(np.nansum([HSP_list[k]*data[k] for k in range(len(data))],0)/np.nansum(HSP_list))
        mu_low.append(np.nansum([(1-HSP_list[k])*data[k] for k in range(len(data))],0)/np.sum([1-hsp for hsp in HSP_list]))
        sigma_high.append(np.nansum([HSP_list[k]*np.matmul(np.array([data[k]-mu_high[-1]]).T,np.array([data[k]-mu_high[-1]])) for k in range(len(data))],0)/np.nansum(HSP_list))
        sigma_low.append(np.nansum([(1-HSP_list[k])*np.matmul(np.array([data[k]-mu_low[-1]]).T,np.array([data[k]-mu_low[-1]])) for k in range(len(data))],0)/np.nansum([1-hsp for hsp in HSP_list]))
        '''
        # update the parameters
        beta.append(np.mean(HSP_list))
        mu_high.append(np.nansum([HSP_list[k]*data[k] for k in range(len(data))],0)/np.nansum(HSP_list))
        mu_low.append(np.nansum([(1-HSP_list[k])*data[k] for k in range(len(data))],0)/np.sum([1-hsp for hsp in HSP_list]))
        sigma_high.append(np.nansum([HSP_list[k]*np.matmul(np.transpose(np.array([data[k]-mu_high[-1]])),np.array([data[k]-mu_high[-1]])) for k in range(len(data))],0)/np.nansum(HSP_list))
        sigma_low.append(np.nansum([(1-HSP_list[k])*np.matmul(np.transpose(np.array([data[k]-mu_low[-1]])),np.array([data[k]-mu_low[-1]])) for k in range(len(data))],0)/np.nansum([1-hsp for hsp in HSP_list]))
        '''
        # update HSP_list      
        for ind_x, x in enumerate(data) :
            try :
                a, b = multivariate_normal.pdf(x,mu_high[-1],sigma_high[-1]), multivariate_normal.pdf(x,mu_low[-1],sigma_low[-1])
            except : # if singular matrix
                a, b = multivariate_normal.pdf(x,mu_high[-1],sigma_high[-1],allow_singular=True), multivariate_normal.pdf(x,mu_low[-1],sigma_low[-1],allow_singular=True)
            HSP_list[ind_x] = (beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
            
        # compute the obtained log-likelihood
        new_LL = 0
        for ind_x, x in enumerate(data) :
            try : 
                if HSP_list[ind_x] >= 0.5 :
                    new_LL -= np.log(beta[-1]) - 0.5*np.log(np.linalg.det(sigma_high[-1])) - 0.5*np.matmul(np.transpose(x-mu_high[-1]),np.matmul(np.linalg.inv(sigma_high[-1]),x-mu_high[-1])) - (nb_timepoints/2)*np.log(2*np.pi)
                else :
                    new_LL -= np.log(1-beta[-1]) - 0.5*np.log(np.linalg.det(sigma_low[-1])) - 0.5*np.matmul(np.transpose(x-mu_low[-1]),np.matmul(np.linalg.inv(sigma_low[-1]),x-mu_low[-1])) - (nb_timepoints/2)*np.log(2*np.pi)
            except : # if singular matrix, which shouldn't happend
                continue
        LLs.append(new_LL)
        
        # assess convergence
        if len(LLs) > 1 :
            if abs(LLs[-2] - LLs[-1]) < 1:
                convergence += 1
            else :
                convergence = 0
            
        i += 1
    
    
    return(convergence>=10, HSP_list, [beta[-1], mu_low[-1], sigma_low[-1], mu_high[-1], sigma_high[-1]], LLs[-1])


def test_GaussianMixture00(data,parameters):
    
    '''
    
    Compute the LL considering the final prediction of the EM algorithm to be true.
    
    Parameters
    ----------
    data : array obtained from SeedSOUNDBETTER.EmpiricalDistribution(SubID,times,preds,conds).
    parameters : len 4 list or array of already fitted parameters.

    Returns
    -------
    neg_LL : float
    average_likelihood : float, average value p(one_point|parameters).

    '''
    
    # unpack the parameters
    beta, mu_low, sigma_low, mu_high, sigma_high = parameters
    
    # compute the log_likelihood
    neg_LL = 0
    for ind, activity in enumerate(data):
        
        # high-state probability (HSP)
        try :
            a, b = multivariate_normal.pdf(activity,mu_high,sigma_high), multivariate_normal.pdf(activity,mu_low,sigma_low)
        except : # if singular covariance matrix
            a, b = multivariate_normal.pdf(activity,mu_high,sigma_high,allow_singular=True), multivariate_normal.pdf(activity,mu_low,sigma_low,allow_singular=True)
        HSP = beta*a/(beta*a + (1-beta)*b)
        
        # likelihood
        if HSP >= 0.5:
            try :
                p_likelihood = multivariate_normal.pdf(activity,mu_high,sigma_high)
            except : # if singular covariance matrix
                p_likelihood = multivariate_normal.pdf(activity,mu_high,sigma_high,allow_singular=True)
        else :
            try :
                p_likelihood = multivariate_normal.pdf(activity,mu_low,sigma_low)
            except : # if singular covariance matrix
                p_likelihood = multivariate_normal.pdf(activity,mu_low,sigma_low,allow_singular=True)
        
        neg_LL -= np.log(p_likelihood)
            
    nb_points = len(data)
    average_likelihood = np.exp(-neg_LL/nb_points)
    
    return(neg_LL, average_likelihood)


def Predict_GaussianMixture00_OldVersion(list_SubIDs, snr, TWOI, nb_iterations=5, cv5=False, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    snr : negative INT
    TWOI : LIST of [2-TUPLE of INT]
    nb_iterations : INT, nb of times the fiting is done for each set of data (with only the best result kept at the end)
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    parameters_names = ['best_beta','best_mu_low','best_sigma_low','best_mu_high','best_sigma_high']
    
    # initiate output
    GaussianMixture00_results = []
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    for SubID in list_SubIDs:
        
        try : 
        
            # Check if it has been done already
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            if os.path.exists('Model9_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
                print('\n Sub'+realSubIDs[SubID]+' already done \n')
                continue
            os.chdir(datapath)
            
            print('\n Start Sub'+realSubIDs[SubID]+'\n')
            
            GaussianMixture00_results_this_sub = {para:[] for para in parameters_names}
            GaussianMixture00_results_this_sub['time'] = list(range(-500,2000,2))
            GaussianMixture00_results_this_sub['lowpassc'] = 10
            GaussianMixture00_results_this_sub['TWOI'] = TWOI
            GaussianMixture00_results_this_sub['exitflags'] = []
            GaussianMixture00_results_this_sub['trainLLH'] = []
            GaussianMixture00_results_this_sub['testLLH'] = []
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            ## cross-validation if asked
            if cv5 :
                for iblocks in range(5):
                    
                    blocks_train = list_of_blocks_train[iblocks]
                    blocks_test = list_of_blocks_test[iblocks]
                    
                    data_list, data_high, data_low = [], [], []
                    for times in TWOI:
                        data_this_time = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
                        data_list.append(data_this_time[snr])
                        data_high.append(data_this_time[-5])
                        data_low.append(data_this_time[-20])
                    data = np.array(data_list).T # transpose so that each row is one subject at all the timepoints
                    data_high = np.array(data_high).T
                    data_low = np.array(data_low).T
                    
                    # compute initial parameters
                    mu_low_guess, mu_high_guess = np.mean(data_low,0), np.mean(data_high,0)
                    sigma_low_guess, sigma_high_guess = np.cov(data_low.T), np.cov(data_high.T)
                    if snr == -9 :
                        beta_guess = 0.5
                    else :
                        beta_guess = 0.75
                
                    iparameters = np.array([beta_guess, mu_low_guess,sigma_low_guess,  mu_high_guess, sigma_high_guess])
                    
                    # train the model
                    try :
                        success, HSP_list, parameters, LL = fit_GaussianMixture00(data,iparameters,maxiter=200*len(iparameters))
                    except TypeError :
                        continue
                    GaussianMixture00_results_this_sub['trainLLH'].append(LL)
                    GaussianMixture00_results_this_sub['exitflags'].append(success)
                    for ind_para, para in enumerate(parameters_names):
                        GaussianMixture00_results_this_sub[para].append(parameters[ind_para])
                    print(success, parameters)
                    print('\n log-likelihood on training set : ', LL) # np.exp(-fun/nb_points))
                    
                    # test the model
                    data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                    try :
                        fun_test, avg_lh = test_GaussianMixture00(data_test,parameters)
                    except TypeError :
                        continue
                    GaussianMixture00_results_this_sub['testLLH'].append(fun_test)
                    print('\n log-likelihood for testing set : ', fun_test, '\n' )
                    
            ## state-prediction on full data
            # real audibility to compare with true state
            snr_to_ind = {-20:1, -13:2, -11:3, -9:4, -7:5, -5:6, -3:7}
            conds_this_snr = conds_this_sub[conds_this_sub.T[4]==snr_to_ind[snr]] # choose just the good snr (4 -> snr-9, 5 -> -7)!!!!
            real_audibility = conds_this_snr.T[0][conds_this_snr.T[1]<21] # remove blocks > 20
            # gather data
            all_data = np.array([EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks=np.linspace(1,20,20))[snr] for times in TWOI]).T
            all_data_low = np.array([EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks=np.linspace(1,20,20))[-20] for times in TWOI]).T
            all_data_high = np.array([EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks=np.linspace(1,20,20))[-5] for times in TWOI]).T
            # initial parameters
            mu_low_guess, mu_high_guess = np.mean(all_data_low,0), np.mean(all_data_high,0)
            sigma_low_guess, sigma_high_guess = np.cov(all_data_low.T), np.cov(all_data_high.T)
            beta_guess = 0.75
            iparameters = [np.random.normal(beta_guess,0.1),[np.random.normal(mu,0.1) for mu in mu_low_guess],sigma_low_guess,[np.random.normal(mu,0.1) for mu in mu_high_guess],sigma_high_guess]
            # train and find HSP_list
            success, HSP_list, parameters, LL = None, None, None, None
            for nb_it in range(nb_iterations):
                try :
                    try_success, try_HSP_list, try_parameters, try_LL = fit_GaussianMixture00(all_data,iparameters,maxiter=200*len(iparameters))
                    if LL == None :
                        success, HSP_list, parameters, LL = try_success, try_HSP_list, try_parameters, try_LL
                    elif try_LL > LL :
                        success, HSP_list, parameters, LL = try_success, try_HSP_list, try_parameters, try_LL
                except TypeError :
                    continue
            if LL == None :
                print('not working for this sub')
                continue
                
            # add tot the output
            GaussianMixture00_results_this_sub['full_data_LLH'] = [LL]
            GaussianMixture00_results_this_sub['full_data_parameters'] = parameters
            GaussianMixture00_results_this_sub['full_data_HSP_list'] = HSP_list
            GaussianMixture00_results_this_sub['full_data_exitflag'] = [success]
            GaussianMixture00_results_this_sub['full_data_correct_prediction'] = []
            for ind_activity, hsp in enumerate(HSP_list) :
                if (hsp >= 0.5 and real_audibility[ind_activity] > 2) or (hsp < 0.5 and real_audibility[ind_activity] <= 2) :
                    GaussianMixture00_results_this_sub['full_data_correct_prediction'].append(1)
                else :
                    GaussianMixture00_results_this_sub['full_data_correct_prediction'].append(0)
            # also add the parameters of the real distributions
            data_real_high = all_data[real_audibility>2]
            data_real_low = all_data[real_audibility<=2]
            real_beta = len(data_real_high)/len(all_data)
            real_mu_high, real_mu_low = np.mean(data_real_high,0), np.mean(data_real_low,0)
            real_sigma_high, real_sigma_low = np.cov(data_real_high.T), np.cov(data_real_low.T)
            GaussianMixture00_results_this_sub['full_data_real_parameters'] = [real_beta, real_mu_high, real_sigma_high, real_mu_low, real_sigma_low]
            real_HSP_list = [] # HSP list given by the real values of the parameters
            for ind, activity in enumerate(all_data):
                try :
                    a, b = multivariate_normal.pdf(activity,real_mu_high,real_sigma_high), multivariate_normal.pdf(activity,real_mu_low,real_sigma_low)
                except : # if singular covariance matrix
                    a, b = multivariate_normal.pdf(activity,real_mu_high,real_sigma_high,allow_singular=True), multivariate_normal.pdf(activity,real_mu_low,real_sigma_low,allow_singular=True)
                real_HSP_list.append(real_beta*a/(real_beta*a + (1-real_beta)*b))
            GaussianMixture00_results_this_sub['full_data_ideal_prediction'] = []
            for ind_activity, hsp in enumerate(real_HSP_list) :
                if (hsp >= 0.5 and real_audibility[ind_activity] > 2) or (hsp < 0.5 and real_audibility[ind_activity] <= 2) :
                    GaussianMixture00_results_this_sub['full_data_ideal_prediction'].append(1)
                else :
                    GaussianMixture00_results_this_sub['full_data_ideal_prediction'].append(0)
            
            # save the results for this subject
            if save :
                
                # average the test_LLH to obtain the model's average LLH
                GaussianMixture00_results_this_sub['GaussianMixture00_LLH'] = np.mean(GaussianMixture00_results_this_sub['testLLH'])
                
                os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
    
                matfile = {}
                for key in GaussianMixture00_results_this_sub.keys():
                    matfile[key] = np.array(GaussianMixture00_results_this_sub[key])
                scipy.io.savemat('Model9_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
                
                os.chdir(datapath)
            
            GaussianMixture00_results.append(GaussianMixture00_results_this_sub)
            
        except ValueError :
            
            print('ValueError with S'+realSubIDs[SubID])
            
            GaussianMixture00_results.append('NaN')
        
    return(GaussianMixture00_results)


def Predict_GaussianMixture00(list_SubIDs, snr, TWOI, nb_iterations=5, aud_thresh=[3],cv5=False, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    snr : negative INT
    TWOI : LIST of [2-TUPLE of INT]
    nb_iterations : INT, nb of times the fiting is done for each set of data (with only the best result kept at the end)
    aud_thresh : LIST of INT. Auditivity thresholds you want to test.
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    parameters_names = ['best_beta','best_mu_low','best_sigma_low','best_mu_high','best_sigma_high']
    
    # initiate output
    GaussianMixture00_results = []
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    for SubID in list_SubIDs:
        
        try : 
        
            # Check if it has been done already
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            if os.path.exists('Model9_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
                print('\n Sub'+realSubIDs[SubID]+' already done \n')
                continue
            os.chdir(datapath)
            
            print('\n Start Sub'+realSubIDs[SubID]+'\n')
            
            GaussianMixture00_results_this_sub = {para:[] for para in parameters_names}
            GaussianMixture00_results_this_sub['time'] = list(range(-500,2000,2))
            GaussianMixture00_results_this_sub['lowpassc'] = 10
            GaussianMixture00_results_this_sub['TWOI'] = TWOI
            GaussianMixture00_results_this_sub['exitflags'] = []
            GaussianMixture00_results_this_sub['trainLLH'] = []
            GaussianMixture00_results_this_sub['testLLH'] = []
            
            preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
            
            ## cross-validation if asked
            if cv5 :
                for iblocks in range(5):
                    
                    blocks_train = list_of_blocks_train[iblocks]
                    blocks_test = list_of_blocks_test[iblocks]
                    
                    data_list, data_high, data_low = [], [], []
                    for times in TWOI:
                        data_this_time = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_train)
                        data_list.append(data_this_time[snr])
                        data_high.append(data_this_time[-5])
                        data_low.append(data_this_time[-20])
                    data = np.array(data_list).T # transpose so that each row is one subject at all the timepoints
                    data_high = np.array(data_high).T
                    data_low = np.array(data_low).T
                    
                    # compute initial parameters
                    mu_low_guess, mu_high_guess = np.mean(data_low,0), np.mean(data_high,0)
                    sigma_low_guess, sigma_high_guess = np.cov(data_low.T), np.cov(data_high.T)
                    if snr == -9 :
                        beta_guess = 0.5
                    else :
                        beta_guess = 0.75
                
                    iparameters = np.array([beta_guess, mu_low_guess,sigma_low_guess,  mu_high_guess, sigma_high_guess])
                    
                    # train the model
                    try :
                        success, HSP_list, parameters, LL = fit_GaussianMixture00(data,iparameters,maxiter=200*len(iparameters))
                    except TypeError :
                        continue
                    GaussianMixture00_results_this_sub['trainLLH'].append(LL)
                    GaussianMixture00_results_this_sub['exitflags'].append(success)
                    for ind_para, para in enumerate(parameters_names):
                        GaussianMixture00_results_this_sub[para].append(parameters[ind_para])
                    print(success, parameters)
                    print('\n log-likelihood on training set : ', LL) # np.exp(-fun/nb_points))
                    
                    # test the model
                    data_test = EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks_test)
                    try :
                        fun_test, avg_lh = test_GaussianMixture00(data_test,parameters)
                    except TypeError :
                        continue
                    GaussianMixture00_results_this_sub['testLLH'].append(fun_test)
                    print('\n log-likelihood for testing set : ', fun_test, '\n' )
                    
            ## state-prediction on full data
            # real audibility to compare with true state
            snr_to_ind = {-20:1, -13:2, -11:3, -9:4, -7:5, -5:6, -3:7}
            conds_this_snr = conds_this_sub[conds_this_sub.T[4]==snr_to_ind[snr]] # choose just the good snr (4 -> snr-9, 5 -> -7)!!!!
            real_audibility = conds_this_snr.T[0][conds_this_snr.T[1]<21] # remove blocks > 20
            # gather data
            all_data = np.array([EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks=np.linspace(1,20,20))[snr] for times in TWOI]).T
            all_data_low = np.array([EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks=np.linspace(1,20,20))[-20] for times in TWOI]).T
            all_data_high = np.array([EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks=np.linspace(1,20,20))[-5] for times in TWOI]).T
            # initial parameters
            mu_low_guess, mu_high_guess = np.mean(all_data_low,0), np.mean(all_data_high,0)
            sigma_low_guess, sigma_high_guess = np.cov(all_data_low.T), np.cov(all_data_high.T)
            beta_guess = 0.75
            iparameters = [beta_guess,mu_low_guess,sigma_low_guess,mu_high_guess,sigma_low_guess]
            #iparameters = [np.random.normal(beta_guess,0.1),[np.random.normal(mu,0.1) for mu in mu_low_guess],sigma_low_guess,[np.random.normal(mu,0.1) for mu in mu_high_guess],sigma_high_guess]
            # train and find HSP_list
            try :
                success, HSP_list, parameters, LL = fit_GaussianMixture00(all_data,iparameters,maxiter=200*len(iparameters))
            except TypeError :
                success, HSP_list, parameters, LL = None, None, None, None, None
            for nb_it in range(nb_iterations-1):
                try :
                    try_iparameters = [np.random.normal(param,0.1) for param in iparameters]
                    try_success, try_HSP_list, try_parameters, try_LL = fit_GaussianMixture00(all_data,try_iparameters,maxiter=200*len(iparameters))
                    if LL == None :
                        success, HSP_list, parameters, LL = try_success, try_HSP_list, try_parameters, try_LL
                    elif try_LL > LL :
                        success, HSP_list, parameters, LL = try_success, try_HSP_list, try_parameters, try_LL
                except TypeError :
                    continue
            if LL == None :
                print('not working for this sub')
                continue
                
            # add tot the output
            GaussianMixture00_results_this_sub['full_data_LLH'] = [LL]
            GaussianMixture00_results_this_sub['full_data_parameters'] = parameters
            GaussianMixture00_results_this_sub['full_data_HSP_list'] = HSP_list
            GaussianMixture00_results_this_sub['full_data_exitflag'] = [success]
            GaussianMixture00_results_this_sub['full_data_correct_prediction'] = {thresh:[] for thresh in aud_thresh}
            for ind_activity, hsp in enumerate(HSP_list) :
                for thresh in aud_thresh :
                    if (hsp >= 0.5 and real_audibility[ind_activity] >= thresh) or (hsp < 0.5 and real_audibility[ind_activity] < thresh) :
                        GaussianMixture00_results_this_sub['full_data_correct_prediction'][thresh].append(1)
                    else :
                        GaussianMixture00_results_this_sub['full_data_correct_prediction'][thresh].append(0)
            # also add the parameters of the real distributions
            data_real_high = {thresh:all_data[real_audibility>=thresh] for thresh in aud_thresh}
            data_real_low = {thresh:all_data[real_audibility<thresh] for thresh in aud_thresh}
            real_beta = {thresh:len(data_real_high[thresh])/len(all_data) for thresh in aud_thresh}
            real_mu_high, real_mu_low = {thresh:np.mean(data_real_high[thresh],0) for thresh in aud_thresh}, {thresh:np.mean(data_real_low[thresh],0) for thresh in aud_thresh}
            real_sigma_high, real_sigma_low = {thresh:np.cov(data_real_high[thresh].T) for thresh in aud_thresh}, {thresh:np.cov(data_real_high[thresh].T) for thresh in aud_thresh}
            GaussianMixture00_results_this_sub['full_data_real_parameters'] = {thresh:[real_beta[thresh], real_mu_high[thresh], real_sigma_high[thresh], real_mu_low[thresh], real_sigma_low[thresh]] for thresh in aud_thresh}
            real_HSP_list = {thresh:[] for thresh in aud_thresh} # HSP list given by the real values of the parameters
            for ind, activity in enumerate(all_data):
                for thresh in aud_thresh :
                    try : 
                        try :
                            a, b = multivariate_normal.pdf(activity,real_mu_high[thresh],real_sigma_high[thresh]), multivariate_normal.pdf(activity,real_mu_low[thresh],real_sigma_low[thresh])
                        except : # if singular covariance matrix
                            a, b = multivariate_normal.pdf(activity,real_mu_high[thresh],real_sigma_high[thresh],allow_singular=True), multivariate_normal.pdf(activity,real_mu_low[thresh],real_sigma_low[thresh],allow_singular=True)
                        real_HSP_list[thresh].append(real_beta[thresh]*a/(real_beta[thresh]*a + (1-real_beta[thresh])*b))
                    except : 
                        print('Issue for thresh '+str(thresh)+', S'+realSubIDs[SubID]+'\n',all)
            GaussianMixture00_results_this_sub['full_data_ideal_prediction'] = {thresh:[] for thresh in aud_thresh}
            for thresh in aud_thresh :
                for ind_activity, hsp in enumerate(real_HSP_list[thresh]) :
                    if (hsp >= 0.5 and real_audibility[ind_activity] >= thresh) or (hsp < 0.5 and real_audibility[ind_activity] < thresh) :
                        GaussianMixture00_results_this_sub['full_data_ideal_prediction'][thresh].append(1)
                    else :
                        GaussianMixture00_results_this_sub['full_data_ideal_prediction'][thresh].append(0)
            
            # save the results for this subject
            if save :
                
                # average the test_LLH to obtain the model's average LLH
                GaussianMixture00_results_this_sub['GaussianMixture00_LLH'] = np.mean(GaussianMixture00_results_this_sub['testLLH'])
                
                os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
    
                matfile = {}
                for key in GaussianMixture00_results_this_sub.keys():
                    matfile[key] = np.array(GaussianMixture00_results_this_sub[key])
                scipy.io.savemat('Model9_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
                
                os.chdir(datapath)
            
            GaussianMixture00_results.append(GaussianMixture00_results_this_sub)
            
        except ValueError :
            
            print('ValueError with S'+realSubIDs[SubID])
            
            GaussianMixture00_results.append('NaN')
        
    return(GaussianMixture00_results)


def Predict_GaussianMixture00_Sklearn(list_SubIDs, snr, TWOI, nb_iterations=5, aud_thresh=[3],cv5=False, save=False, redo=False):
    
    '''
    Parameters
    ----------
    list_SubIDs : LIST of INT. Indexes of the subjects in realSubIDs (so between 0 and 19).
    snr : negative INT
    TWOI : LIST of [2-TUPLE of INT]
    nb_iterations : INT, nb of times the fiting is done for each set of data (with only the best result kept at the end)
    aud_thresh : LIST of INT. Auditivity thresholds you want to test.
    save :BOOL , optional. If True, save the results.

    Returns
    -------
    Bifurcation00_results : LIST of DICT. Results of the CV5 for each subject.

    '''
    # get data
    datapath = seedSOUNDBETTER.datapath
    os.chdir(datapath)
    file_name = os.path.join('Subject_All_Active',
                             'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
    classif = scipy.io.loadmat(file_name)
    preds, conds = classif['preds'][0], classif['conds'][0]
    
    # initiate output
    GaussianMixture00_results = []
    
    # function that will be used to fit the model
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return(-estimator.bic(X))
    
    for SubID in list_SubIDs:
        
        # Check if it has been done already
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model9_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        os.chdir(datapath)
        
        print('\n Start Sub'+realSubIDs[SubID]+'\n')
        
        # initialize output
        GaussianMixture00_results_this_sub = {}
        
        # real audibility to compare with true state
        preds_this_sub, conds_this_sub = preds[SubID], conds[SubID]
        snr_to_ind = {-20:1, -13:2, -11:3, -9:4, -7:5, -5:6, -3:7}
        conds_this_snr = conds_this_sub[conds_this_sub.T[4]==snr_to_ind[snr]] # choose just the good snr (4 -> snr-9, 5 -> -7)!!!!
        real_audibility = conds_this_snr.T[0][conds_this_snr.T[1]<21] # remove blocks > 20
        # gather activity data
        all_data = np.array([EmpiricalDistribution(times,preds_this_sub,conds_this_sub,blocks=np.linspace(1,20,20))[snr] for times in TWOI]).T
        
        # check the hyperparameters
        param_grid = {"n_components": range(1, 5),"covariance_type": ["tied","full"],}
        grid_search = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score)
        grid_search.fit(all_data)
        GaussianMixture00_results_this_sub['best_hyperparams'] = grid_search.best_params_
        print('\n nb of mixtures for S'+realSubIDs[SubID]+' : ',grid_search.best_params_['n_components'])
        
        # fit with hyperparameters (2,'full')
        GMM = GaussianMixture(n_components=2, covariance_type='full')
        GMM.fit(all_data)
        predictions = GMM.predict(all_data)
        GaussianMixture00_results_this_sub['predictions'] = predictions
        
        # fill-in the correct predictions
        GaussianMixture00_results_this_sub['correct_actual_predictions'] = {thresh:[] for thresh in aud_thresh}
        for ind_activity, p in enumerate(predictions) :
            for thresh in aud_thresh :
                if (p==1 and real_audibility[ind_activity] >= thresh) or (p==0 and real_audibility[ind_activity] < thresh) :
                    GaussianMixture00_results_this_sub['correct_actual_prediction'][thresh].append(1)
                else :
                    GaussianMixture00_results_this_sub['correct_actual_prediction'][thresh].append(0)
                    
        # also add the parameters of the real distributions and the ideal predictions
        data_real_high = {thresh:all_data[real_audibility>=thresh] for thresh in aud_thresh}
        data_real_low = {thresh:all_data[real_audibility<thresh] for thresh in aud_thresh}
        real_beta = {thresh:len(data_real_high[thresh])/len(all_data) for thresh in aud_thresh}
        real_mu_high, real_mu_low = {thresh:np.mean(data_real_high[thresh],0) for thresh in aud_thresh}, {thresh:np.mean(data_real_low[thresh],0) for thresh in aud_thresh}
        real_sigma_high, real_sigma_low = {thresh:np.cov(data_real_high[thresh].T) for thresh in aud_thresh}, {thresh:np.cov(data_real_high[thresh].T) for thresh in aud_thresh}
        GaussianMixture00_results_this_sub['real_parameters'] = {thresh:[real_beta[thresh], real_mu_high[thresh], real_sigma_high[thresh], real_mu_low[thresh], real_sigma_low[thresh]] for thresh in aud_thresh}
        real_HSP_list = {thresh:[] for thresh in aud_thresh} # HSP list given by the real values of the parameters
        for ind, activity in enumerate(all_data):
            for thresh in aud_thresh :
                try : 
                    try :
                        a, b = multivariate_normal.pdf(activity,real_mu_high[thresh],real_sigma_high[thresh]), multivariate_normal.pdf(activity,real_mu_low[thresh],real_sigma_low[thresh])
                    except : # if singular covariance matrix
                        a, b = multivariate_normal.pdf(activity,real_mu_high[thresh],real_sigma_high[thresh],allow_singular=True), multivariate_normal.pdf(activity,real_mu_low[thresh],real_sigma_low[thresh],allow_singular=True)
                    real_HSP_list[thresh].append(real_beta[thresh]*a/(real_beta[thresh]*a + (1-real_beta[thresh])*b))
                except : 
                    print('Issue for thresh '+str(thresh)+', S'+realSubIDs[SubID]+'\n',all)
        GaussianMixture00_results_this_sub['correct_ideal_prediction'] = {thresh:[] for thresh in aud_thresh}
        for thresh in aud_thresh :
            for ind_activity, hsp in enumerate(real_HSP_list[thresh]) :
                if (hsp >= 0.5 and real_audibility[ind_activity] >= thresh) or (hsp < 0.5 and real_audibility[ind_activity] < thresh) :
                    GaussianMixture00_results_this_sub['correct_ideal_prediction'][thresh].append(1)
                else :
                    GaussianMixture00_results_this_sub['correct_ideal_prediction'][thresh].append(0)
        
        # save the results for this subject
        if save :
            
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')

            matfile = {}
            for key in GaussianMixture00_results_this_sub.keys():
                matfile[key] = np.array(GaussianMixture00_results_this_sub[key])
            scipy.io.savemat('Model9_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat', matfile)
            
            os.chdir(datapath)
        
        GaussianMixture00_results.append(GaussianMixture00_results_this_sub)
        
    return(GaussianMixture00_results)

