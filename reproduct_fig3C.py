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
        audibility = audibility_array[epo_ind]
        if audibility in [0,1,2] :
            iaud = 0
        elif audibility in [3,4,5,6,7,8,9,10]:
            iaud = 1
        
        # post-onset distribution
        average_value = np.mean(epo[int(0.8*sfreq):int(0.9*sfreq)])
        empirical_distribution_this_sub_350ms[snr][iaud].append(average_value)
        
    # add this distribution to the main dict
    empirical_distributions_350ms[SubID] = empirical_distribution_this_sub_350ms
        
    

#%% Plot Fig 3C for each subject

print('\n ... plots for each sub ...\n')

for SubID in range(20):
    
    data_this_sub = empirical_distributions_350ms[SubID]
    
    heards = []
    not_heards = []
    
    for snr in SNRs:
        
        data_this_snr = data_this_sub[snr]
        heards.append(np.mean(data_this_snr[1]))
        not_heards.append(np.mean(data_this_snr[0]))
    
    plt.figure()
    plt.title('Fig3C Sub'+realSubIDs[SubID])
    plt.plot(SNRs[1:],heards[1:],color='orange',marker='o')
    plt.plot(SNRs[1:],not_heards[1:],color='violet',marker='o')
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Sanity_checks')
    plt.savefig('FIG_3C_reprod '+realSubIDs[SubID])
    os.chdir(datapath)
    plt.close()
    
    
#%% Plot Fig 3C for eall subjects

print('\n ... plots for average ...\n')

heards = {snr : [] for snr in SNRs}
not_heards = {snr : [] for snr in SNRs}

for SubID in range(20):
    
    data_this_sub = empirical_distributions_350ms[SubID]
    
    for snr in SNRs:
        
        data_this_snr = data_this_sub[snr]
        heards[snr].append(np.nanmean(data_this_snr[1]))
        not_heards[snr].append(np.nanmean(data_this_snr[0]))
    
heards = [np.nanmean(heards[snr]) for snr in SNRs]
not_heards = [np.nanmean(not_heards[snr]) for snr in SNRs]
    
plt.figure()
plt.title('Fig3C All Subjects')
plt.plot(SNRs[1:],heards[1:],color='orange',marker='o')
plt.plot(SNRs[1:],not_heards[1:],color='violet',marker='o')
os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Sanity_checks')
plt.savefig('FIG_3C_reprod_all_subjects')
os.chdir(datapath)

        
    