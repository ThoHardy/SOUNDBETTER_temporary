# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 22:44:06 2023

@author: thoma
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
import seedSOUNDBETTER
import scipy.io
import os

realSubIDs = seedSOUNDBETTER.SubIDs

datapath = seedSOUNDBETTER.datapath

mat2mne = seedSOUNDBETTER.mat2mne

os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/TFA')

redo=0

for SubID in range(20):   
    
    if os.path.exists('S'+realSubIDs[SubID]+'_alphaOz700ms_VS_SCPFCz300ms.png')==1 and redo==0 :
        print('S'+realSubIDs[SubID]+' already done')
        continue 
    
    if realSubIDs[SubID]=='17':
        continue
    
    print('Process sub'+realSubIDs[SubID]+'...')
    
    # already-made EpochsTFR files
    ''' 
    os.chdir(datapath+'/myPowers_Active')
    powers_heard = mne.time_frequency.read_tfrs('Power_'+realSubIDs[SubID]+'_5-tfr.h5')[0].pick(['Cz','P3'])['audibility > 2'] #.apply_baseline((-0.4,-0.1),mode='ratio')
    powers_not_heard = mne.time_frequency.read_tfrs('Power_'+realSubIDs[SubID]+'_5-tfr.h5')[0].pick(['Cz','P3'])['audibility < 3'] #.apply_baseline((-0.4,-0.1),mode='ratio')
    '''
    
    # in-place made EpochsTFR objects
    data_ref = datapath + '/' + 'Subject_' + realSubIDs[SubID] + '_Active' + '/data_ref.mat'
    myEpoch = mat2mne(data_ref, True).pick(['Cz','C1','C2','CP1','CP2','CPz','O1','Oz','O2','PO1','POz','PO2'])
    freqs = np.arange(1,14,1)
    tfr_epochs = mne.time_frequency.tfr_morlet(myEpoch,freqs,n_cycles=freqs/2,decim=2,
                                               average=False,return_itc=False) # tfa
    tfr_epochs = tfr_epochs['snr == 4']
    tfr_epochs.apply_baseline(mode='logratio',baseline=(-0.5,-0.1)) # normalize
    powers_heard, powers_not_heard = tfr_epochs['audibility > 2'], tfr_epochs['audibility < 3']
    powers_heard_scp, powers_not_heard_scp = powers_heard.copy().pick(['Cz','C1','C2','CP1','CP2','CPz']).crop(tmin=0.2,tmax=0.5,fmin=1,fmax=5)._data, powers_not_heard.copy().pick(['Cz','C1','C2','CP1','CP2','CPz']).crop(tmin=0.2,tmax=0.5,fmin=1,fmax=5)._data
    powers_heard_alpha, powers_not_heard_alpha = powers_heard.copy().pick(['O1','Oz','O2','PO1','POz','PO2']).crop(tmin=0.5,tmax=0.75,fmin=8,fmax=12)._data, powers_not_heard.copy().pick(['O1','Oz','O2','PO1','POz','PO2']).crop(tmin=0.5,tmax=0.75,fmin=8,fmax=12)._data
    
    positions_heard_scp, positions_not_heard_scp = [np.mean(trial) for trial in powers_heard_scp], [np.mean(trial) for trial in powers_not_heard_scp]
    positions_heard_alpha, positions_not_heard_alpha = [np.mean(trial) for trial in powers_heard_alpha], [np.mean(trial) for trial in powers_not_heard_alpha]
        
    positions_heard = np.array([positions_heard_scp,positions_heard_alpha]).T
    positions_not_heard = np.array([positions_not_heard_scp,positions_not_heard_alpha]).T
    
    
    linreg_heard = np.polyfit(positions_heard[:,0],positions_heard[:,1],1,full=True)
    a_heard, b_heard, r_heard = linreg_heard[0][0], linreg_heard[0][1], linreg_heard[1][0]/len(positions_heard)
    linreg_not_heard = np.polyfit(positions_not_heard[:,0],positions_not_heard[:,1],1,full=True)
    a_not_heard, b_not_heard, r_not_heard = linreg_not_heard[0][0], linreg_not_heard[0][1], linreg_not_heard[1][0]/len(positions_not_heard)
        
    
    os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/TFA')
    plt.figure()
    
    for trial in positions_heard:
        plt.scatter(trial[0],trial[1],color='blue')
    for trial in positions_not_heard:
        plt.scatter(trial[0],trial[1],color='red' )
        
    plt.plot([-2,2],[-2*a_heard+b_heard,2*a_heard+b_heard],color='blue',label='a='+str(int(1000*a_heard)/1000)+', r='+str(int(1000*r_heard)/1000))
    plt.plot([-2,2],[-2*a_not_heard+b_not_heard,2*a_not_heard+b_not_heard],color='red',label='a='+str(int(1000*a_not_heard)/1000)+', r='+str(int(1000*r_not_heard)/1000))
        
    plt.xlabel('SCP')
    plt.ylabel('alpha')
    plt.title('separability visualisation trial on powers (alpha and scp, different channels and times)')
    plt.legend()
    plt.savefig('S'+realSubIDs[SubID]+'_alphaOz700ms_VS_SCPFCz300ms.png')
    plt.close()
    
# last participant done in the usual way : S09.