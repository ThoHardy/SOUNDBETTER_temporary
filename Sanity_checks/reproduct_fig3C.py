# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:25:28 2023

@author: thoma
"""

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



#%% Create an array with the (snr*projcted activity) distributions for each subject

empirical_distributions_350ms = {}

for SubID in range(20) :
    
    print('\n ... begin with Subject'+realSubIDs[SubID]+' ...\n')
    
    # initialize variables containing the measurements of interest
    empirical_distribution_this_sub_350ms = {x:[[],[]] for x in SNRs}
    
    sfreq = 500
    
    # select the data for just this subject
    preds_this_sub = preds[0][SubID]
    
    # get the metadata
    snr_array = conds[0][SubID][:,4]
    audibility_array = conds[0][SubID][:,0]
    
    # fill in the empirical distribution dicts WITH average on 30ms (15*2ms)
    for epo_ind, epo in enumerate(preds_this_sub):
        
        snr = ind_to_snr[snr_array[epo_ind]]
        audibility = int(ind_to_snr[audibility_array[epo_ind]]>5)
        
        # post-onset distribution
        average_value = np.mean(epo[int(0.8*sfreq)-8:int(0.9*sfreq)+7])
        empirical_distribution_this_sub_350ms[snr][audibility].append(average_value)
        
    # add this distribution to the main dict
    empirical_distributions_350ms[SubID] = empirical_distribution_this_sub_350ms
        
    

#%% Find the mean and std of the baseline activity for each subject

print('\n ... plots ...\n')

for SubID in range(20):
    
    data_this_sub = empirical_distributions_350ms[SubID]
    
    heards = []
    not_heards = []
    
    for snr in SNRs:
        
        data_this_snr = data_this_sub[snr]
        heards.append(np.mean(data_this_snr[1]))
        not_heards.append(np.mean(data_this_snr[0]))
    
    plt.figure()
    plt.title('Fig3C Sub'+SubID)
    plt.plot(SNRs[1:],heards[1:],color='orange',marker='o')
    plt.plot(SNRs[1:],not_heards[1:],color='violet',marker='o')
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')
    plt.savefig('FIG_3C_reprod '+realSubIDs[SubID])
    os.chdir(datapath)
    plt.close()
    
    
        
    