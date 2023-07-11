# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:41:12 2023

@author: thoma
"""

import scipy.io
import os
import seedSOUNDBETTER
import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.optimize import minimize 
from scipy.stats import multivariate_normal
import time

datapath = seedSOUNDBETTER.datapath
realSubIDs = seedSOUNDBETTER.SubIDs


#%% Gather data from clusters of electrodes, on specified blocks of trials

def GatherData(Epoch, channels_list, tmin, tmax, blocks):
    
    '''
    Parameters
    ----------
    Epoch : mne.Epoch object from one subject.
    channels_list : LIST of LIST of STRING, shape (nb_clusters)*(nb_channels).
    blocks : ARRAY of INT.

    Returns
    -------
    data : array of shape (nb_trials)*(nb_clusters)
    metadata : dataframe of shape (nb_trials)*(6)

    '''
    
    myEpoch = Epoch['blocknumber in '+str(list(blocks))]
    myEpoch.crop(tmin=tmin,tmax=tmax)
    
    data = np.array([np.mean(myEpoch.copy().pick(clust_chan)._data.T,1) for clust_chan in channels_list]).T # shape (nb_trials)*(nb_times)*(nb_clusters)
    
    data_averaged_in_time = np.array([np.mean(trial,0) for trial in data]) # shape (nb_trials)*(nb_clusters)
    
    return(data_averaged_in_time,myEpoch.metadata)



#%% Fit Unimodal model on (data, metadata)

def FitUnimodal(data,metadata,maxiter=1000):
    
    '''
    Parameters
    ----------
    data : ARRAY of shape (nb_trials)*(nb_clusters) obtained with GatherData().
    metadata : DATAFRAME of shape (nb_trials)*(6) obtained with GatherData().
    maxiter : INT, maximum number of iterations before giving up the convergence of the optimization method.

    Returns
    -------
    success : BOOL, True if convergence.
    parameters : ARRAY-like of FLOAT, optimal parameters.
    fun : FLOAT, maximum LL.

    '''
    
    # nb of dimensions of the data
    nb_dims = data.shape[1]
    
    # compute initial parameters
    snrs = np.array(metadata['snr'])
    data_silence, data_loud = data[snrs==1], data[snrs==6]
    slope_sigma_guess_total = (np.cov(data_loud.T) - np.cov(data_silence.T))/5
    slope_sigma_guess = np.array([slope_sigma_guess_total[i,i] for i in range(len(slope_sigma_guess_total))]) # extract diagonal
    intercept_sigma_guess = np.cov(data_silence.T) - np.diag(slope_sigma_guess) # approx_sigma(0dB)
    k_guess = (np.mean(data_loud,0) - np.mean(data_silence,0))/8
    threshsnr_guess = 4
    mu_maxsnr_guess = np.mean(data_loud,0)
    L_guess = 2*mu_maxsnr_guess
    
    # function to minimize
    def MLE_unimodal(parameters):
        # extract parameters
        k = parameters[:nb_dims]
        mu_maxsnr = parameters[nb_dims:2*nb_dims]
        L = parameters[2*nb_dims:3*nb_dims]
        slope_sigma = parameters[3*nb_dims:4*nb_dims]
        intercept_sigma = np.array([parameters[(k+4)*nb_dims:(k+5)*nb_dims] for k in range(nb_dims)])
        threshsnr = parameters[-1]
        
        # compute the log_likelihood
        neg_LL = 0
        for indx, x in enumerate(data):
            snr = snrs[indx]
            mu_snr = L/(1+np.exp(-k*(snr-threshsnr))) - L/(1+np.exp(-k*(-5-threshsnr))) + mu_maxsnr
            sigma_snr = np.diag(slope_sigma*mu_snr) + intercept_sigma
            p_likelihood = multivariate_normal.pdf(x,mu_snr,sigma_snr)
            neg_LL -= np.log(p_likelihood)
            
        return(neg_LL)
    
    # minimize MLE_unimodal in the parameters space
    initial_params = np.array(list(k_guess)+list(mu_maxsnr_guess)+list(L_guess)+list(slope_sigma_guess)+list(intercept_sigma_guess.flatten())+[threshsnr_guess])
    try : 
        mle_model = minimize(MLE_unimodal, 
                         initial_params,
                         method='Nelder-Mead',
                         options = {'return_all':False,'maxfev':maxiter,'maxiter':maxiter})
        return(mle_model['success'], mle_model['x'], -mle_model['fun'])
    except ValueError : # occurs if sigma is not positive semidefinite
        return(0,np.nan,np.nan)
        


#%% Compute the LL 
    
def TestUnimodal(data, metadata, parameters):
    
    '''
    Parameters
    ----------
    data : ARRAY of shape (nb_trials)*(nb_clusters) obtained with GatherData().
    metadata : DATAFRAME of shape (nb_trials)*(6) obtained with GatherData().
    parameters : ARRAY-like of FLOAT, optimal parameters on the training set. 

    Returns
    -------
    LL : log_likelihood computed on the given dataset. 

    '''
    
    # snrs
    snrs = np.array(metadata['snr'])
    
    # nb of dimensions of the data
    nb_dims = data.shape[1]
    
    # extract parameters
    k = parameters[:nb_dims]
    mu_maxsnr = parameters[nb_dims:2*nb_dims]
    L = parameters[2*nb_dims:3*nb_dims]
    slope_sigma = parameters[3*nb_dims:4*nb_dims]
    intercept_sigma = np.array([parameters[(k+4)*nb_dims:(k+5)*nb_dims] for k in range(nb_dims)])
    threshsnr = parameters[-1]
    
    # compute the log_likelihood
    LL = 0
    for indx, x in enumerate(data):
        snr = snrs[indx]
        mu_snr = L/(1+np.exp(-k*(snr-threshsnr))) - L/(1+np.exp(-k*(-5-threshsnr))) + mu_maxsnr
        sigma_snr = np.diag(slope_sigma*mu_snr) + intercept_sigma
        p_likelihood = multivariate_normal.pdf(x,mu_snr,sigma_snr)
        LL += np.log(p_likelihood)
        
    return(LL)



#%% Fit Bimodal model on (data, metadata)

def FitBimodal(data,metadata,maxiter=1000):
    
    '''
    Parameters
    ----------
    data : ARRAY of shape (nb_trials)*(nb_clusters) obtained with GatherData().
    metadata : DATAFRAME of shape (nb_trials)*(6) obtained with GatherData().
    maxiter : INT, maximum number of iterations before giving up the convergence of the optimization method.

    Returns
    -------
    success : BOOL, True if convergence.
    parameters : ARRAY-like of FLOAT, optimal parameters.
    fun : FLOAT, maximum LL.

    '''
    
    # nb of dimensions of the data
    nb_dims = data.shape[1]
    
    # compute initial parameters
    snrs = np.array(metadata['snr'])
    data_silence, data_loud = data[snrs==1], data[snrs==6]
    mu_silence_guess, sigma_guess = np.mean(data_silence,0), np.cov(data_silence.T)
    L_high_guess = np.mean(data_loud,0) - np.mean(data_silence,0)
    k_guess, k_high_guess = np.mean(L_high_guess)/5, L_high_guess/5
    step_guess = np.mean(data[snrs==2],0) - np.mean(data_silence)
    threshsnr_guess = 4
    
    # function to minimize
    def MLE_bimodal(parameters):
        # extract parameters
        mu_silence = parameters[:nb_dims]
        L_high = parameters[nb_dims:2*nb_dims]
        k_high = parameters[2*nb_dims:3*nb_dims]
        step = parameters[3*nb_dims:4*nb_dims]
        sigma = np.array([parameters[(k+4)*nb_dims:(k+5)*nb_dims] for k in range(nb_dims)])
        [k, threshsnr] = parameters[-2:]
        
        # compute the log_likelihood
        neg_LL = 0
        for indx, x in enumerate(data):
            snr = snrs[indx]
            beta_snr = 1/(1+np.exp(-k*(snr-threshsnr)))
            mu_snr = step + np.array([L_high[i]/(1+np.exp(-k_high[i]*(snr-threshsnr))) for i in range(nb_dims)])
            p_likelihood = beta_snr*multivariate_normal.pdf(x,mu_snr,sigma) + (1-beta_snr)*multivariate_normal.pdf(x,mu_silence,sigma)
            neg_LL -= np.log(p_likelihood)
            
        return(neg_LL)
    
    # minimize MLE_unimodal in the parameters space
    initial_params = np.array(list(mu_silence_guess)+list(L_high_guess)+list(k_high_guess)+list(step_guess)+list(sigma_guess.flatten())+[k_guess,threshsnr_guess])
    try :
        mle_model = minimize(MLE_bimodal, 
                         initial_params,
                         method='Nelder-Mead',
                         options = {'return_all':False,'maxfev':maxiter,'maxiter':maxiter})
        return(mle_model['success'], mle_model['x'], -mle_model['fun'])
    except ValueError :
        return(0,np.nan,np.nan)
    
    

#%% Compute the LL 
    
def TestBimodal(data, metadata, parameters):
    
    '''
    Parameters
    ----------
    data : ARRAY of shape (nb_trials)*(nb_clusters) obtained with GatherData().
    metadata : DATAFRAME of shape (nb_trials)*(6) obtained with GatherData().
    parameters : ARRAY-like of FLOAT, optimal parameters on the training set. 

    Returns
    -------
    LL : log_likelihood computed on the given dataset. 

    '''
    
    # snrs
    snrs = np.array(metadata['snr'])
    
    # nb of dimensions of the data
    nb_dims = data.shape[1]
    
    # extract parameters
    mu_silence = parameters[:nb_dims]
    L_high = parameters[nb_dims:2*nb_dims]
    k_high = parameters[2*nb_dims:3*nb_dims]
    step = parameters[3*nb_dims:4*nb_dims]
    sigma = np.array([parameters[(k+4)*nb_dims:(k+5)*nb_dims] for k in range(nb_dims)])
    [k, threshsnr] = parameters[-2:]
    
    # compute the log_likelihood
    LL = 0
    for indx, x in enumerate(data):
        snr = snrs[indx]
        beta_snr = 1/(1+np.exp(-k*(snr-threshsnr)))
        mu_snr = step + np.array([L_high[i]/(1+np.exp(-k_high[i]*(snr-threshsnr))) for i in range(nb_dims)])
        p_likelihood = beta_snr*multivariate_normal.pdf(x,mu_snr,sigma) + (1-beta_snr)*multivariate_normal.pdf(x,mu_silence,sigma)
        LL += np.log(p_likelihood)
        
    return(LL)



#%% CV on any model

def CV5(model, SubIDs, channels_list, times_array='all', maxiter=300, redo=0):
    
    # model <-> gather, fit and train functions
    models2gather = {'7MV':GatherData,'2BMV':GatherData}
    models2fit = {'7MV':FitBimodal,'2BMV':FitUnimodal}
    models2test = {'7MV':TestBimodal,'2BMV':TestUnimodal}
    if not(model in models2fit.keys()):
        raise ValueError('"model" unknown')
    Gather, Fit, Test = models2gather[model], models2fit[model], models2test[model]
    
    # create the training and testing blocks
    list_of_blocks_train = [np.linspace(5,20,16),
                      np.concatenate((np.linspace(1,4,4),np.linspace(9,20,12))),
                      np.concatenate((np.linspace(1,8,8),np.linspace(13,20,8))),
                      np.concatenate((np.linspace(1,12,12),np.linspace(17,20,4))),
                      np.linspace(1,16,16)]
    list_of_blocks_test = [np.linspace(1,4,4),np.linspace(5,8,4),np.linspace(9,12,4),
                           np.linspace(13,16,4),np.linspace(17,20,4)]
    
    # set initial time values
    if times_array == 'all':
        times = np.linspace(-0.3,1.8,53)
    else :
        times = times_array
    
    # initialize global output
    AVGtestLL = []
    
    # CV5 for each subject
    for SubID in SubIDs :
        
        # check if already done for this subject
        os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
        if os.path.exists('Model'+model+'_Twind_5cv_active_S'+realSubIDs[SubID]+'.mat')==1 and redo==0:
            print('\n Sub'+realSubIDs[SubID]+' already done \n')
            continue
        
        print('\n Start S'+realSubIDs[SubID]+'... \n')
        
        # load epoch file
        if model in ['2BMV','7MV']: # raw channels signal
            os.chdir(datapath + '/myEpochs_Active')
            data_ref = 'Epoch_'+realSubIDs[SubID]+'-epo.fif'
            Epoch = mne.read_epochs(data_ref, preload=True)
        
        # initialize outputs
        success_list, trainLL_list, testLL_list, AVGtestLL_list, params_list = [], [], [], [], []
        
        # for each timepoint
        for t in times:
            
            print('\n S'+realSubIDs[SubID]+', '+str(t)+'s... \n')
            
            success_list_this_time, trainLL_list_this_time, testLL_list_this_time, params_list_this_time = [], [], [], []
            
            # for each fold
            for fold in range(5):
                
                print('\n Fold nb'+str(fold)+'...')
                
                # train
                start = time.time()
                data, metadata = Gather(Epoch.copy(), channels_list, 
                                            tmin=t, tmax=t+0.03, 
                                            blocks=list_of_blocks_train[fold])
                success, params, trainLL = Fit(data,metadata,maxiter=maxiter)
                end = time.time()
                print('Training complete ('+str(end-start)[:5]+'s)')
                print('Convergence :', success)
                
                # test
                if not(isinstance(params,np.ndarray)): # if params if np.nan
                    print('ValueError during training !')
                else : 
                    start = time.time()
                    data, metadata = Gather(Epoch.copy(), channels_list, 
                                                tmin=t, tmax=t+0.03, 
                                                blocks=list_of_blocks_test[fold])
                    testLL = Test(data, metadata, params)
                    end = time.time()
                    print('Testing complete ('+str(end-start)[:5]+'s)')
                
                # store results for this fold
                success_list_this_time.append(int(success))
                trainLL_list_this_time.append(trainLL)
                testLL_list_this_time.append(testLL)
                params_list_this_time.append(params)
                
            # store results for this timepoint
            success_list.append(success_list_this_time)
            trainLL_list.append(trainLL_list_this_time)
            testLL_list.append(testLL_list_this_time)
            AVGtestLL_list.append(np.nanmean(testLL_list_this_time))
            params_list.append(params_list_this_time)
            
        # store results for this subject
        AVGtestLL.append(AVGtestLL_list)
        if times == 'all' :
            os.chdir(datapath+'/ModelComparisonResults_Twind30ms_Thomas')
            matfile = {'LLH':AVGtestLL_list, 'exitflags':success_list, 'trainLLH':trainLL, 'testLLH':testLL,
                       'channels_clusters':channels_list,'params':params_list}
            scipy.io.savemat('Model'+model+'_Twind_5cv_active_S'+realSubIDs[SubID]+'.mat', matfile)
        
    return(AVGtestLL)

#%% Set some variables

SubIDs = range(20)

channels_list = [['F1','Fz','F2','FC1','FCz','FC2'],['FT7','FC5','T7','C5','TP7','CP5'],
                 ['FT8','FC6','T8','C6','TP8','CP6'],['PO3','POz','PO4','O1','Oz','O2']]

#channels_list = [['F1','Fz','F2','FC1','FCz','FC2'],['C5','C3','C1','CP5','CP3','CP1'],
                 #['C6','C4','C2','CP6','CP4','CP2'],['PO3','POz','PO4','O1','Oz','O2']]

    
    
    
    

    
    
    
    
    
    
    