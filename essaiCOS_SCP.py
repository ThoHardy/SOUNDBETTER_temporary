# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:55:49 2023

@author: thoma
"""

import scipy.io
import os
import seedSOUNDBETTER
import matplotlib.pyplot as plt
import numpy as np
import mne

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

realSubIDs = seedSOUNDBETTER.SubIDs


datapath = 'D:/SOUNDBETTER_Thomas/ModelComparisonResults_Twind30ms_Thomas'


uni, bi = '13', '12'
list_unimodal, list_bimodal = [], []

for SubID in range(20):
    
    os.chdir(datapath)
    LLH_unimodal = scipy.io.loadmat('Model'+uni+'_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')['LLH']
    LLH_bimodal = scipy.io.loadmat('Model'+bi+'_Twind_5cv_active_filtered_10Hz_S'+realSubIDs[SubID]+'.mat')['LLH']
    
    list_unimodal.append(LLH_unimodal)
    list_bimodal.append(LLH_bimodal)
    
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/comp_LLH')
    plt.figure()
    plt.plot(2*np.array(range(101,901,15))[:-1]-500,LLH_unimodal[0]-LLH_bimodal[0],color='c',label='difference')
    #plt.plot(2*np.array(range(101,901,15))[:-1]-500,-LLH_unimodal[0],color='m',label='unimodal')
    #plt.plot(2*np.array(range(101,901,15))[:-1]-500,-LLH_bimodal[0],color='m',label='bimodal')
    plt.title('Unimodal VS bimodal GMMs, on COS, S'+realSubIDs[SubID])
    plt.xlabel('time')
    plt.ylabel('LLH')
    plt.legend()
    plt.savefig('diff_COS_S'+realSubIDs[SubID]+'.png')
    plt.close()


avg_unimodal = np.mean(np.array(list_unimodal),0)
avg_bimodal = np.mean(np.array(list_bimodal),0)

plt.figure()
plt.plot(2*np.array(range(101,901,15))[:-1]-500,-avg_unimodal[0],color='c')
plt.plot(2*np.array(range(101,901,15))[:-1]-500,-avg_bimodal[0],color='m')
plt.show()

plt.figure()
plt.plot(2*np.array(range(101,901,15))[:-1]-500,-avg_bimodal[0]+avg_unimodal[0],color='c')
plt.show()

    
    
    
    