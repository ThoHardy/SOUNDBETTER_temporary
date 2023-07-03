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

os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/TFA')

redo=1

for SubID in range(20):   
    
    if os.path.exists('S'+realSubIDs[SubID]+'_alphaOz700ms_VS_SCPFCz300ms.png')==1 and redo==0 :
        print('S'+realSubIDs[SubID]+' already done')
        continue 
    
    if realSubIDs[SubID]=='17':
        continue

    os.chdir(datapath+'/myPowers_Active')
    powers_heard = mne.time_frequency.read_tfrs('Power_'+realSubIDs[SubID]+'_5-tfr.h5')[0].pick(['Cz','P3'])['audibility > 2'] #.apply_baseline((-0.4,-0.1),mode='ratio')
    powers_not_heard = mne.time_frequency.read_tfrs('Power_'+realSubIDs[SubID]+'_5-tfr.h5')[0].pick(['Cz','P3'])['audibility < 3'] #.apply_baseline((-0.4,-0.1),mode='ratio')
    
    powers_heard, powers_not_heard = powers_heard.data, powers_not_heard.data
    
    positions_heard, positions_not_heard = [[np.mean(trial[0,1:4,390:410]),np.mean(trial[1,8:13,490:510])] for trial in powers_heard], [[np.mean(trial[0,1:4,390:410]),np.mean(trial[1,8:13,590:610])] for trial in powers_not_heard]
        
    positions_heard = np.array(positions_heard)
    positions_not_heard = np.array(positions_not_heard)
    
    
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