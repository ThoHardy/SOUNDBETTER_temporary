#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:02:04 2017

@author: clairesergent
"""

# Extract trial by trial classification scores at each time point
# Based on a JR King example JR_example_classification_single_trial_score.py

#%%
#matplotlib inline
from os import path as op
from scipy.io import savemat
import numpy as np # library for matrix operations
import mne
import matplotlib.pyplot as plt  # library for plotting

datapath = '/Volumes/Lacie/SoundConscious/'

#%%

def mat2mne(file_name,active):
    # converts the matlab fieldtrip file with name file_name to mne format
    # active: is it an active or passive session (true or false)
    
    import scipy.io as sio
    from mne.epochs import EpochsArray
    from pandas import DataFrame
    
    # Read channel struture from mne example
    rawdata_path = '/Volumes/Lacie/SoundConscious/'
    raw_fname = rawdata_path + 'template_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.pick_types(eeg=True)
    
    # Convert Claire's data from Fieldtrip structure to mne Epochs
    mat = sio.loadmat(file_name,squeeze_me=True, struct_as_record=False)
    
    ft_data = mat['data_ref']
    n_trial = len(ft_data.trial)
    n_chans, n_time = ft_data.trial[0].shape
    data = np.zeros((n_trial, n_chans, n_time))
    for trial in range(n_trial):
        data[trial, :, :] = ft_data.trial[trial]
    sfreq = float(ft_data.fsample)  # sampling frequency
    coi = range(63)  # channels of interest:
    data = data[:, coi, :]
    info = raw.info
    info['sfreq'] = sfreq
    
    #1 : blocknumber
    #2 : snr (1=no sound ; 2= -13dB ; 3=-11dB ; 4=-9dB : 5=-7dB ; 6 =-5dB)
    #3 : vowel (1=A 2=E)
    #4-active : eval (0=incorrect vowel response ; 1=correct vowel response)
    #4-passive : questiontype (1=Quiz 2=RT 3=MindWander 4=None)
    #5 : respside (1=left ; 2=right)
    #6-active : audibility [0-10]
    #6-passive : response [1-4]
    
     
    y = dict()
    
    if active==True:   
        for ii, name in enumerate(['blocknumber', 'snr', 'vowel', 'eval', 'respside', 'audibility']):
            y[name] = ft_data.trialinfo[:, ii]
    else:
        for ii, name in enumerate(['blocknumber', 'snr', 'vowel', 'questiontype', 'respside', 'response']):
            y[name] = ft_data.trialinfo[:, ii]
    
    y = DataFrame(y)
    
    # make MNE epochs, assuming one event every 5 seconds
    events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                   np.zeros(n_trial), np.zeros(n_trial)].astype(int)
    epochs = EpochsArray(data, info, events=events, tmin=ft_data.time[0][0],
                         verbose=False)
    return epochs, y

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ACTIVE SESSION : SINGLE TRIAL PREDICTIONS AND SCORES, GENERALIZATION FROM -5dB 
# TO OTHER SNRs AND GENERALIZATION IN TIME, ALSO SCORING SEPARATELY HEARD AND NOT HEARD
# 
#   The generalization conditions (intermediate SNRs) are scored with the same procedure 
#   as the training conditions (no sound and max SNR) 
#   
#   Same as previous version below, but with GENERALIZATION IN TIME (by changing SlidingEstimator to GeneralizingEstimator),
#   DECIMATION 5 and ONE FILE PER SUBJECT
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from mne.decoding import GeneralizingEstimator

subjects_list = ['01','02','03','05','06','07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']
snr_dB = ['NoSound','m13dB','m11dB','m9dB','m7dB','m5dB']; # CHANGE FOR PASSIVE TEN LAST SUBJECTS

#CHANGE PARAMETERS HERE 
active = 1 # 0 or 1
n_cvsplits = 10 # number of cross validation splits
# DECIMATION FACTOR HERE
dfactor = 5

# setup logistic regression classifier
clf = make_pipeline(
    StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
    LogisticRegression(),
)
sliding = GeneralizingEstimator(clf,scoring='roc_auc', n_jobs=-1)

for subject in subjects_list:

    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    epochs.decimate(dfactor)
    X = epochs.get_data()
    n_trials = X.shape[0]
    n_times = X.shape[2]
    snr = y['snr']
    
    # Find indices of minimum and max snr, with which we'll train the
    # classifiers
    train_cond = np.where((y['snr'] == 1) | (y['snr'] == 6))[0]
    # Find indices of the other intermediary snr trials
    gen_cond = np.where((y['snr'] > 1) & (y['snr'] < 6))[0]
    
    # Setup a unique cross validation scheme, applied separately for
    # the training and generalization conditions
    cv = StratifiedKFold(n_splits=n_cvsplits, random_state=0)
    
    # Apply cross-validation scheme on training and generalization sets
    cv_train = cv.split(X[train_cond], snr[train_cond])
    cv_gen = cv.split(X[gen_cond], np.ones_like(snr[gen_cond]))
    
    # Retrieve corresponding indices
    trains, tests = zip(*[(train_cond[train], train_cond[test])
                          for train, test in cv_train])
    gens = [gen_cond[test] for _, test in cv_gen]
    
    # Cross-validation loop for single trial predictions
    y_pred = np.zeros((n_trials, n_times, n_times))
    scores_fold = np.zeros((5,n_times,n_times,n_cvsplits)) # ATTENTION: CHANGE IF MAX SNR + -3dB THEN 6 SNRs
    scores_heard_fold = np.zeros((5,n_times,n_times,n_cvsplits)) 
    scores_notheard_fold = np.zeros((5,n_times,n_times,n_cvsplits)) 
    
    for fold, (train, test, gen) in enumerate(zip(trains, tests, gens)):
        # Check that train on 0 and max snr
        assert set(snr[train]) == {1, 6}
        # Fit    
        sliding.fit(X=X[train], y=snr[train])
        # Predict maxSNR and no sound 
        y_pred[test] = sliding.decision_function(X[test])
        # Generalize to other SNRs
        y_pred[gen] = sliding.decision_function(X[gen])
        
        for i, isnr in enumerate(range(2,6)): # from SNR 2 to 5
            idx = np.r_[test[snr[test]==1], gen[snr[gen]==isnr]]
            scores_fold[i,...,fold] = sliding.score(X=X[idx],y=snr[idx])
        
        scores_fold[4,...,fold] = sliding.score(X=X[test],y=snr[test])
    
    scores = np.mean(scores_fold,axis=3)
    
    plt.figure()
    plt.matshow(np.mean(y_pred[snr==6],axis=0),origin = 'lower', vmin=-1, vmax=1, cmap='RdBu_r')
    
    #    plt.figure
    #    itimes = (epochs.times >0) & (epochs.times < 0.5) # (period for averaging the prediciton)
    #    std = np.zeros(6)
    #    for this_snr in range(6):
    #        std[this_snr] = np.std(np.mean(y_pred[snr==this_snr+1][:,itimes],1))
    #    plt.plot(std)
    
    
    print(fname)
    
    preds = np.array(y_pred)
    conds = np.array(y)    
    conds_order = '1:audibility  2:blocknumber  3:eval  4:respside  5:snr  6:vowel'
    
    scores = np.array(scores)
    scores_dimnames = 'snr 2 to 6, n_times, n_times'
    
    if active==True: 
        savemat(datapath + 'Subject_' + subject + '_Active/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})
    else:
        savemat(datapath + 'Subject_' + subject + '_Passive/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})
    



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PASSIVE SESSION LAST TEN SUBJECTS : SINGLE TRIAL PREDICTIONS AND SCORES, GENERALIZATION FROM -3dB 
# TO OTHER SNRs AND GENERALIZATION IN TIME
# 
#   The generalization conditions (intermediate SNRs) are scored with the same procedure 
#   as the training conditions (no sound and max SNR) 
#   
#   Same as previous version below, but with GENERALIZATION IN TIME (by changing SlidingEstimator to GeneralizingEstimator),
#   DECIMATION 5 and ONE FILE PER SUBJECT
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from mne.decoding import GeneralizingEstimator

subjects_list = ['14','15','17','19','20','22','23','24','25'] # '13' done alone for test
snr_dB = ['NoSound','m13dB','m11dB','m9dB','m7dB','m5dB','m3dB']; # CHANGE FOR PASSIVE TEN LAST SUBJECTS

#CHANGE PARAMETERS HERE 
active = 0 # 0 or 1
n_cvsplits = 10 # number of cross validation splits
# DECIMATION FACTOR HERE
dfactor = 5

# setup logistic regression classifier
clf = make_pipeline(
    StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
    LogisticRegression(),
)
sliding = GeneralizingEstimator(clf,scoring='roc_auc', n_jobs=-1)

for subject in subjects_list:

    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    epochs.decimate(dfactor)
    X = epochs.get_data()
    n_trials = X.shape[0]
    n_times = X.shape[2]
    snr = y['snr']
    
    # Find indices of minimum and max snr, with which we'll train the
    # classifiers
    train_cond = np.where((y['snr'] == 1) | (y['snr'] == 7))[0]
    # Find indices of the other intermediary snr trials
    gen_cond = np.where((y['snr'] > 1) & (y['snr'] < 7))[0]
    
    # Setup a unique cross validation scheme, applied separately for
    # the training and generalization conditions
    cv = StratifiedKFold(n_splits=n_cvsplits, random_state=0)
    
    # Apply cross-validation scheme on training and generalization sets
    cv_train = cv.split(X[train_cond], snr[train_cond])
    cv_gen = cv.split(X[gen_cond], np.ones_like(snr[gen_cond]))
    
    # Retrieve corresponding indices
    trains, tests = zip(*[(train_cond[train], train_cond[test])
                          for train, test in cv_train])
    gens = [gen_cond[test] for _, test in cv_gen]
    
    # Cross-validation loop for single trial predictions
    y_pred = np.zeros((n_trials, n_times, n_times))
    scores_fold = np.zeros((6,n_times,n_times,n_cvsplits)) # IF MAX SNR + -3dB THEN 6 SNRs
    
    for fold, (train, test, gen) in enumerate(zip(trains, tests, gens)):
        # Check that train on 0 and max snr
        assert set(snr[train]) == {1, 7}
        # Fit    
        sliding.fit(X=X[train], y=snr[train])
        # Predict maxSNR and no sound 
        y_pred[test] = sliding.decision_function(X[test])
        # Generalize to other SNRs
        y_pred[gen] = sliding.decision_function(X[gen])
        
        for i, isnr in enumerate(range(2,7)): # from SNR 2 to 6
            idx = np.r_[test[snr[test]==1], gen[snr[gen]==isnr]]
            scores_fold[i,...,fold] = sliding.score(X=X[idx],y=snr[idx])
        
        scores_fold[5,...,fold] = sliding.score(X=X[test],y=snr[test])
    
    scores = np.mean(scores_fold,axis=3)
    
    #plt.figure()
    #plt.matshow(np.mean(y_pred[snr==6],axis=0),origin = 'lower', vmin=-1, vmax=1, cmap='RdBu_r')
    
    #    plt.figure
    #    itimes = (epochs.times >0) & (epochs.times < 0.5) # (period for averaging the prediciton)
    #    std = np.zeros(6)
    #    for this_snr in range(6):
    #        std[this_snr] = np.std(np.mean(y_pred[snr==this_snr+1][:,itimes],1))
    #    plt.plot(std)
    
    
    print(fname)
    
    preds = np.array(y_pred)
    conds = np.array(y)    
    conds_order = '1:audibility  2:blocknumber  3:eval  4:respside  5:snr  6:vowel'
    
    scores = np.array(scores)
    scores_dimnames = 'snr 2 to 6, n_times, n_times'
    
    if active==True: 
        savemat(datapath + 'Subject_' + subject + '_Active/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})
    else:
        savemat(datapath + 'Subject_' + subject + '_Passive/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PASSIVE SESSION FIRST TEN SUBJECTS : SINGLE TRIAL PREDICTIONS AND SCORES, GENERALIZATION FROM -5dB 
# TO OTHER SNRs AND GENERALIZATION IN TIME
# 
#   The generalization conditions (intermediate SNRs) are scored with the same procedure 
#   as the training conditions (no sound and max SNR) 
#   
#   Same as previous version below, but with GENERALIZATION IN TIME (by changing SlidingEstimator to GeneralizingEstimator),
#   DECIMATION 5 and ONE FILE PER SUBJECT
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from mne.decoding import GeneralizingEstimator

subjects_list = ['02','03','05','06','07','08','09','11','12']
snr_dB = ['NoSound','m13dB','m11dB','m9dB','m7dB','m5dB']; # CHANGE FOR PASSIVE TEN LAST SUBJECTS

#CHANGE PARAMETERS HERE 
active = 0 # 0 or 1
n_cvsplits = 10 # number of cross validation splits
# DECIMATION FACTOR HERE
dfactor = 5

# setup logistic regression classifier
clf = make_pipeline(
    StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
    LogisticRegression(),
)
sliding = GeneralizingEstimator(clf,scoring='roc_auc', n_jobs=-1)

for subject in subjects_list:

    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    epochs.decimate(dfactor)
    X = epochs.get_data()
    n_trials = X.shape[0]
    n_times = X.shape[2]
    snr = y['snr']
    
    # Find indices of minimum and max snr, with which we'll train the
    # classifiers
    train_cond = np.where((y['snr'] == 1) | (y['snr'] == 6))[0]
    # Find indices of the other intermediary snr trials
    gen_cond = np.where((y['snr'] > 1) & (y['snr'] < 6))[0]
    
    # Setup a unique cross validation scheme, applied separately for
    # the training and generalization conditions
    cv = StratifiedKFold(n_splits=n_cvsplits, random_state=0)
    
    # Apply cross-validation scheme on training and generalization sets
    cv_train = cv.split(X[train_cond], snr[train_cond])
    cv_gen = cv.split(X[gen_cond], np.ones_like(snr[gen_cond]))
    
    # Retrieve corresponding indices
    trains, tests = zip(*[(train_cond[train], train_cond[test])
                          for train, test in cv_train])
    gens = [gen_cond[test] for _, test in cv_gen]
    
    # Cross-validation loop for single trial predictions
    y_pred = np.zeros((n_trials, n_times, n_times))
    scores_fold = np.zeros((5,n_times,n_times,n_cvsplits)) # ATTENTION: CHANGE IF MAX SNR + -3dB THEN 6 SNRs
    
    for fold, (train, test, gen) in enumerate(zip(trains, tests, gens)):
        # Check that train on 0 and max snr
        assert set(snr[train]) == {1, 6}
        # Fit    
        sliding.fit(X=X[train], y=snr[train])
        # Predict maxSNR and no sound 
        y_pred[test] = sliding.decision_function(X[test])
        # Generalize to other SNRs
        y_pred[gen] = sliding.decision_function(X[gen])
        
        for i, isnr in enumerate(range(2,6)): # from SNR 2 to 5
            idx = np.r_[test[snr[test]==1], gen[snr[gen]==isnr]]
            scores_fold[i,...,fold] = sliding.score(X=X[idx],y=snr[idx])
        
        scores_fold[4,...,fold] = sliding.score(X=X[test],y=snr[test])
    
    scores = np.mean(scores_fold,axis=3)
    
    #plt.figure()
    #plt.matshow(np.mean(y_pred[snr==6],axis=0),origin = 'lower', vmin=-1, vmax=1, cmap='RdBu_r')
    
    #    plt.figure
    #    itimes = (epochs.times >0) & (epochs.times < 0.5) # (period for averaging the prediciton)
    #    std = np.zeros(6)
    #    for this_snr in range(6):
    #        std[this_snr] = np.std(np.mean(y_pred[snr==this_snr+1][:,itimes],1))
    #    plt.plot(std)
    
    
    print(fname)
    
    preds = np.array(y_pred)
    conds = np.array(y)    
    conds_order = '1:audibility  2:blocknumber  3:eval  4:respside  5:snr  6:vowel'
    
    scores = np.array(scores)
    scores_dimnames = 'snr 2 to 6, n_times, n_times'
    
    if active==True: 
        savemat(datapath + 'Subject_' + subject + '_Active/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})
    else:
        savemat(datapath + 'Subject_' + subject + '_Passive/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ACTIVE SESSION : SINGLE TRIAL PREDICTIONS AND SCORES, GENERALIZATION FROM -5dB 
# TO OTHER SNRs AND GENERALIZATION IN TIME
# 
#   The generalization conditions (intermediate SNRs) are scored with the same procedure 
#   as the training conditions (no sound and max SNR) 
#   
#   Same as previous version below, but with GENERALIZATION IN TIME (by changing SlidingEstimator to GeneralizingEstimator),
#   DECIMATION 5 and ONE FILE PER SUBJECT
#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from mne.decoding import GeneralizingEstimator

subjects_list = ['07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']
#subjects_list = ['01','02','03','05','06','07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']
snr_dB = ['NoSound','m13dB','m11dB','m9dB','m7dB','m5dB']; # CHANGE FOR PASSIVE TEN LAST SUBJECTS

#CHANGE PARAMETERS HERE 
active = 1 # 0 or 1
n_cvsplits = 10 # number of cross validation splits
# DECIMATION FACTOR HERE
dfactor = 5

# setup logistic regression classifier
clf = make_pipeline(
    StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
    LogisticRegression(),
)
sliding = GeneralizingEstimator(clf,scoring='roc_auc', n_jobs=-1)

for subject in subjects_list:

    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    epochs.decimate(dfactor)
    X = epochs.get_data()
    n_trials = X.shape[0]
    n_times = X.shape[2]
    snr = y['snr']
    
    # Find indices of minimum and max snr, with which we'll train the
    # classifiers
    train_cond = np.where((y['snr'] == 1) | (y['snr'] == 6))[0]
    # Find indices of the other intermediary snr trials
    gen_cond = np.where((y['snr'] > 1) & (y['snr'] < 6))[0]
    
    # Setup a unique cross validation scheme, applied separately for
    # the training and generalization conditions
    cv = StratifiedKFold(n_splits=n_cvsplits, random_state=0)
    
    # Apply cross-validation scheme on training and generalization sets
    cv_train = cv.split(X[train_cond], snr[train_cond])
    cv_gen = cv.split(X[gen_cond], np.ones_like(snr[gen_cond]))
    
    # Retrieve corresponding indices
    trains, tests = zip(*[(train_cond[train], train_cond[test])
                          for train, test in cv_train])
    gens = [gen_cond[test] for _, test in cv_gen]
    
    # Cross-validation loop for single trial predictions
    y_pred = np.zeros((n_trials, n_times, n_times))
    scores_fold = np.zeros((5,n_times,n_times,n_cvsplits)) # ATTENTION: CHANGE IF MAX SNR + -3dB THEN 6 SNRs
    
    for fold, (train, test, gen) in enumerate(zip(trains, tests, gens)):
        # Check that train on 0 and max snr
        assert set(snr[train]) == {1, 6}
        # Fit    
        sliding.fit(X=X[train], y=snr[train])
        # Predict maxSNR and no sound 
        y_pred[test] = sliding.decision_function(X[test])
        # Generalize to other SNRs
        y_pred[gen] = sliding.decision_function(X[gen])
        
        for i, isnr in enumerate(range(2,6)): # from SNR 2 to 5
            idx = np.r_[test[snr[test]==1], gen[snr[gen]==isnr]]
            scores_fold[i,...,fold] = sliding.score(X=X[idx],y=snr[idx])
        
        scores_fold[4,...,fold] = sliding.score(X=X[test],y=snr[test])
    
    scores = np.mean(scores_fold,axis=3)
    
    plt.figure()
    plt.matshow(np.mean(y_pred[snr==6],axis=0),origin = 'lower', vmin=-1, vmax=1, cmap='RdBu_r')
    
    #    plt.figure
    #    itimes = (epochs.times >0) & (epochs.times < 0.5) # (period for averaging the prediciton)
    #    std = np.zeros(6)
    #    for this_snr in range(6):
    #        std[this_snr] = np.std(np.mean(y_pred[snr==this_snr+1][:,itimes],1))
    #    plt.plot(std)
    
    
    print(fname)
    
    preds = np.array(y_pred)
    conds = np.array(y)    
    conds_order = '1:audibility  2:blocknumber  3:eval  4:respside  5:snr  6:vowel'
    
    scores = np.array(scores)
    scores_dimnames = 'snr 2 to 6, n_times, n_times'
    
    if active==True: 
        savemat(datapath + 'Subject_' + subject + '_Active/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})
    else:
        savemat(datapath + 'Subject_' + subject + '_Passive/classif_SingleTrialPred_and_AUC_vowelpresence_train_maxSNR_test_allSNR_decimate%i' % dfactor + '.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order, 'scores' : scores, 'scores_dimnames' : scores_dimnames})
    



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SINGLE TRIAL PREDICTIONS WITHOUT TEMPORAL GENERALIZATION (NO DECIMATION)
#   The generalization conditions (intermediate SNRs) are scored with the same procedure 
#   as the training conditions (no sound and max SNR) 

#%% 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator

subjects_list = ['01','02','03','05','06','07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']
snr_dB = ['NoSound','m13dB','m11dB','m9dB','m7dB','m5dB'];

#CHANGE PARAMETERS HERE 
active = 0 # 0 or 1
n_cvsplits = 10 # number of cross validation splits

# setup logistic regression classifier
clf = make_pipeline(
    StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
    LogisticRegression(),
)
sliding = SlidingEstimator(clf, n_jobs=-1)

all_trlspred = list()
all_trlsconds = list()
count = 0

for subject in subjects_list:

    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    X = epochs.get_data()
    n_trials = X.shape[0]
    n_times = X.shape[2]
    snr = y['snr']
    
    # Find indices of minimum and max snr, with which we'll train the
    # classifiers
    train_cond = np.where((y['snr'] == 1) | (y['snr'] == 6))[0]
    # Find indices of the other intermediary snr trials
    gen_cond = np.where((y['snr'] > 1) & (y['snr'] < 6))[0]
    
    # Setup a unique cross validation scheme, applied separately for
    # the training and generalization conditions
    cv = StratifiedKFold(n_splits=n_cvsplits, random_state=0)
    
    # Apply cross-validation scheme on training and generalization sets
    cv_train = cv.split(X[train_cond], snr[train_cond])
    cv_gen = cv.split(X[gen_cond], np.ones_like(snr[gen_cond]))
    
    # Retrieve corresponding indices
    trains, tests = zip(*[(train_cond[train], train_cond[test])
                          for train, test in cv_train])
    gens = [gen_cond[test] for _, test in cv_gen]
    
    # Cross-validation loop for single trial predictions
    y_pred = np.zeros((n_trials, n_times))
    for (train, test, gen) in zip(trains, tests, gens):
        # Check that train on 0 and max snr
        assert set(snr[train]) == {1, 6}
        # Fit    
        sliding.fit(X=X[train], y=snr[train])
        # Predict
        y_pred[test] = sliding.decision_function(X[test])
        # Generalize
        y_pred[gen] = sliding.decision_function(X[gen])
    
#    plt.figure()
#    plt.plot(epochs.times,np.mean(y_pred[snr==6],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==5],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==4],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==3],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==2],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==1],axis=0))
#    plt.axhline(0, color='k')
#    plt.ylabel('mean score')
#    plt.xlabel('Times (ms)')
#    
#    plt.figure
#    itimes = (epochs.times >0) & (epochs.times < 0.5) # (period for averaging the prediciton)
#    std = np.zeros(6)
#    for this_snr in range(6):
#        std[this_snr] = np.std(np.mean(y_pred[snr==this_snr+1][:,itimes],1))
#    plt.plot(std)
    
    all_trlspred.append(y_pred)
    all_trlsconds.append(np.array(y))
    count = count + 1
    print(fname)

preds = np.array(all_trlspred)
conds = np.array(all_trlsconds)

conds_order = '1:audibility  2:blocknumber  3:eval  4:respside  5:snr  6:vowel'

if active==True: 
    savemat(datapath + '/Subject_All_Active/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_%i' % count  + '_subjects.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order})
else:
    savemat(datapath + '/Subject_All_Passive/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_passive_%i' % count + '_subjects.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order})

  
#%% ANALYSIS WHERE MAXSNR = -3 dB IN THE PASSIVE SESSION, restricted to the ten last subjects

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator

subjects_list = ['13','14','15','17','19','20','22','23','24','25']
snr_dB = ['NoSound','m13dB','m11dB','m9dB','m7dB','m5dB','m3dB'];

#CHANGE PARAMETERS HERE 
active = 0 # 0 or 1
n_cvsplits = 10 # number of cross validation splits

# setup logistic regression classifier
clf = make_pipeline(
    StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
    LogisticRegression(),
)
sliding = SlidingEstimator(clf, n_jobs=-1)

all_trlspred = list()
all_trlsconds = list()
count = 0

for subject in subjects_list:

    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    X = epochs.get_data()
    n_trials = X.shape[0]
    n_times = X.shape[2]
    snr = y['snr']
    
    # Find indices of minimum and max snr, with which we'll train the
    # classifiers
    train_cond = np.where((y['snr'] == 1) | (y['snr'] == 7))[0]
    # Find indices of the other intermediary snr trials
    gen_cond = np.where((y['snr'] > 1) & (y['snr'] < 7))[0]
    
    # Setup a unique cross validation scheme, applied separately for
    # the training and generalization conditions
    cv = StratifiedKFold(n_splits=n_cvsplits, random_state=0)
    
    # Apply cross-validation scheme on training and generalization sets
    cv_train = cv.split(X[train_cond], snr[train_cond])
    cv_gen = cv.split(X[gen_cond], np.ones_like(snr[gen_cond]))
    
    # Retrieve corresponding indices
    trains, tests = zip(*[(train_cond[train], train_cond[test])
                          for train, test in cv_train])
    gens = [gen_cond[test] for _, test in cv_gen]
    
    # Cross-validation loop for single trial predictions
    y_pred = np.zeros((n_trials, n_times))
    for (train, test, gen) in zip(trains, tests, gens):
        # Check that train on 0 and max snr
        assert set(snr[train]) == {1, 7}
        # Fit    
        sliding.fit(X=X[train], y=snr[train])
        # Predict
        y_pred[test] = sliding.decision_function(X[test])
        # Generalize
        y_pred[gen] = sliding.decision_function(X[gen])
    
#    plt.figure()
#    plt.plot(epochs.times,np.mean(y_pred[snr==6],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==5],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==4],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==3],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==2],axis=0))
#    plt.plot(epochs.times,np.mean(y_pred[snr==1],axis=0))
#    plt.axhline(0, color='k')
#    plt.ylabel('mean score')
#    plt.xlabel('Times (ms)')
#    
#    plt.figure
#    itimes = (epochs.times >0) & (epochs.times < 0.5) # (period for averaging the prediciton)
#    std = np.zeros(6)
#    for this_snr in range(6):
#        std[this_snr] = np.std(np.mean(y_pred[snr==this_snr+1][:,itimes],1))
#    plt.plot(std)
    
    all_trlspred.append(y_pred)
    all_trlsconds.append(np.array(y))
    count = count + 1
    print(fname)

preds = np.array(all_trlspred)
conds = np.array(all_trlsconds)

conds_order = '1:audibility  2:blocknumber  3:eval  4:respside  5:snr  6:vowel'

if active==True: 
    savemat(datapath + '/Subject_All_Active/classif_SingleTrialPred_vowelpresence_train_maxSNRm3sB_test_allSNR_active_%i' % count  + '_subjects.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order})
else:
    savemat(datapath + '/Subject_All_Passive/classif_SingleTrialPred_vowelpresence_train_maxSNRm3dB_test_allSNR_passive_%i' % count + '_subjects.mat', {'preds': preds, 'conds' : conds, 'conds_order' : conds_order})

   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
#   PREVIOUS VERSION (before 22 Sept 2017): the generalization conditions (intermediate SNRs) where not scored with the same procedure 
#   as the training conditions (no sound and max SNR) which artificially increased the standard deviation across 
#   trials for the training conditions: indeed, for the training conditions the pred was estimated based on training 
#   on a subsample of the trials which changed for each cv split, which was not the case for the generalisation 
#   (no cv split).

#%% Generalize of classification of target presence from maximal SNR to another SNR 
# Just Timedecoding then extract "score" on individual trials

# setup logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from mne.decoding import SlidingEstimator, cross_val_multiscore

subjects_list = ['01','02','03','05','06','07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']
snr_dB = ['NoSound','m13dB','m11dB','m9dB','m7dB','m5dB'];

#CHANGE PARAMETERS HERE 
active = 0 # 0 or 1
snr = 5 # the sound level condition to be scored, BE CAREFUL : 'no sound' = 1

all_trlscores = list()
count = 0

for subject in subjects_list:
        
    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    
    # specify the classifier 
    clf = make_pipeline(
        StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        LogisticRegression(),
    )
    
    # This will apply the logistic on each time sample separately
    sliding = SlidingEstimator(clf, scoring='roc_auc')
    
    # select the trials to fit and the trials to score
    sel_fit = np.where((y['snr'] == 1) | (y['snr'] == 6))[0]
    sel_score = np.where((y['snr'] == snr))[0]
        
    y_fit = np.array(y.loc[sel_fit, 'snr'])
    y_fit[y_fit == 1] = 0
    y_fit[y_fit > 1] = 1
    
    sliding.fit(X=epochs[sel_fit].get_data(),y=y_fit)
    
    y_pred = sliding.predict_proba(X=epochs[sel_score].get_data())
    
    # the proba are for each class, since it's only two classes, we can skip one colum
    y_pred = y_pred[:,:,1] # take column 1
    
    all_trlscores.append(y_pred)
    count = count + 1
    print(fname)
    
    # Plot single trial prediction
#    plt.matshow(y_pred,
#                cmap='RdBu_r', vmin=0., vmax=1.)
#    plt.xlabel('Time')
#    plt.ylabel('Trials')
#    plt.colorbar()
#    
    plt.figure()
    plt.plot(epochs.times,np.mean(y_pred,axis=0))
    plt.axhline(0.5, color='k')
    plt.ylabel('mean score')
    plt.xlabel('Times (ms)')
#    
#    plt.figure()
#    plt.plot(epochs.times,np.var(y_pred,axis=0))
#    plt.ylabel('Var score')
#    plt.xlabel('Times (ms)')

scores = np.array(all_trlscores)
if active==True: 
    savemat(datapath + '/Subject_All_Active/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_SNR' + snr_dB[snr-1] + '_active_%i' % count  + '_subjects.mat', {'scores': scores})
else:
    savemat(datapath + '/Subject_All_Passive/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_SNR' + snr_dB[snr-1] + '_passive_%i' % count + '_subjects.mat', {'scores': scores})


#%% Single trial prediction on vowel presence, max SNR versus noise

# setup logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from mne.decoding import SlidingEstimator, cross_val_multiscore

subjects_list = ['01','02','03','05','06','07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']

#CHANGE PARAMETERS HERE 
active = 0 # 0 or 1

all_preds_nosound = list()
all_preds_maxsnr = list()

count = 0

for subject in subjects_list:
    
    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    
    # select the trials to fit and score 
    sel = np.where((y['snr'] == 1) | (y['snr'] == 6))[0]
        
    y = np.array(y.loc[sel, 'snr'])
    y[y == 1] = 0
    y[y > 1] = 1
    
    X = epochs[sel].get_data()
        
    ntimes = X.shape[-1]
    
    #Loop over timepoints
    Y_pred = list()
    for t in range(0,ntimes,1):
        #select a specific timepoint
        Xt = X[:,:,t]         
        y_pred = cross_val_predict(LogisticRegression(), Xt, y, cv=10, method='predict_proba')  
        y_pred = y_pred[:,1] # take column 1
        Y_pred.append(y_pred)
        
    preds = np.array(Y_pred)
    preds = preds.transpose()
    
    preds_nosound = preds[y == 0]
    preds_maxsnr = preds[y == 1]
    
    plt.figure()
    plt.plot(epochs.times,np.mean(preds_nosound,axis=0))
    plt.plot(epochs.times,np.mean(preds_maxsnr,axis=0))
    plt.legend(['NoSound','MaxSNR'])
    
    plt.figure()
    plt.plot(epochs.times,np.std(preds_nosound,axis=0))
    plt.plot(epochs.times,np.std(preds_maxsnr,axis=0))
    plt.legend(['NoSound','MaxSNR'])
    
    # close all figures: plt.close('all')
    
    all_preds_nosound.append(preds_nosound)
    all_preds_maxsnr.append(preds_maxsnr)

    count = count + 1
    print(fname)

scores_nosound = np.array(all_preds_nosound)
scores_maxsnr = np.array(all_preds_maxsnr)
if active==True: 
    savemat(datapath + '/Subject_All_Active/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_NoSound_active_%i' % count  + '_subjects.mat', {'scores': scores_nosound})
    savemat(datapath + '/Subject_All_Active/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_maxSNR_active_%i' % count  + '_subjects.mat', {'scores': scores_maxsnr})
else:
    savemat(datapath + '/Subject_All_Passive/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_NoSound_passive_%i' % count  + '_subjects.mat', {'scores': scores_nosound})
    savemat(datapath + '/Subject_All_Passive/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_maxSNR_passive_%i' % count  + '_subjects.mat', {'scores': scores_maxsnr})





#%%
from mne.decoding import SlidingEstimator, cross_val_multiscore, cross_val_predict

X = epochs.get_data()
y = epochs.events[:, 2]
y_pred = cross_val_predict(LogisticRegression(), X, y, cv=10, method='predict_proba')
accuracy = lambda proba, category: proba * (category==0) + (1-proba) * (category==1)
y_score = accuracy(y_pred, y)






#%% Generalize of classification of target presence from maximal SNR to another SNR : -9 dB
# Just Timedecoding then extract "score" on individual trials

# setup logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from mne.decoding import SlidingEstimator, cross_val_multiscore

subjects_list = ['01']#,'02','03','05','06','07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']

#for active in [0,1]:
active = 1

all_trlscores = list()
count = 0

for subject in subjects_list:
        
    #load data
    if active==True:
        fname = op.join(datapath, 'Subject_' + subject +'_Active', 'data_ref.mat')
    else:
        fname = op.join(datapath, 'Subject_' + subject +'_Passive', 'data_ref.mat')
    
    epochs, y = mat2mne(fname,1)
    
    # specify the classifier 
    clf = make_pipeline(
        StandardScaler(), # Z-score data, because gradiometers and magnetometers have different scales
        LogisticRegression(),
    )
    
    # This will apply the logistic on each time sample separately
    sliding = SlidingEstimator(clf, scoring='roc_auc')
    
    # select the trials to fit and the trials to score
    sel_fit = np.where((y['snr'] == 1) | (y['snr'] == 6))[0]
    sel_score = np.where((y['snr'] == 4))[0]
        
    y_fit = np.array(y.loc[sel_fit, 'snr'])
    y_fit[y_fit == 1] = 0
    y_fit[y_fit > 1] = 1
    
    sliding.fit(X=epochs[sel_fit].get_data(),y=y_fit)
    
    y_pred = sliding.predict_proba(X=epochs[sel_score].get_data())
    
    # the proba are for each class, since it's only two classes, we can skip one colum
    y_pred = y_pred[:,:,0]
    
    all_trlscores.append(y_pred)
    count = count + 1
    print(fname)
    
    # Plot single trial prediction
#    plt.matshow(y_pred,
#                cmap='RdBu_r', vmin=0., vmax=1.)
#    plt.xlabel('Time')
#    plt.ylabel('Trials')
#    plt.colorbar()
#    
#    plt.figure()
#    plt.plot(epochs.times,np.mean(y_pred,axis=0))
#    plt.axhline(0.5, color='k')
#    plt.ylabel('mean score')
#    plt.xlabel('Times (ms)')
#    
#    plt.figure()
#    plt.plot(epochs.times,np.var(y_pred,axis=0))
#    plt.ylabel('Var score')
#    plt.xlabel('Times (ms)')

scores = np.array(all_trlscores)
if active==True: 
    savemat(datapath + '/Subject_All_Active/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_SNRm9dB_active_%i' % count  + '_subjects.mat', {'scores': scores})
else:
    savemat(datapath + '/Subject_All_Passive/classif_SingleTrialPred_vowelpresence_train_maxSNR_test_SNRm9dB_passive_%i' % count + '_subjects.mat', {'scores': scores})


