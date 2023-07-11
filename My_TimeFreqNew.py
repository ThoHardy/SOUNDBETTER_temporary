# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:09:36 2023

@author: thoma
"""

import os
import mne
import scipy.io as sio
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from pandas import DataFrame
from pathlib import Path
from random import randrange
import h5py
import matplotlib as mpl
import seedSOUNDBETTER

# specify working directory
datapath=seedSOUNDBETTER.datapath
realSubIDs = seedSOUNDBETTER.SubIDs
os.chdir(datapath)


#%% Compute and save TFA

def calcPower(SubIDs, ActOrPass):
    
    '''
    Adapted from Martina's script.
    '''
    
    # epoArrlist: obtained using loadEpochs
    # cond = conditions of interest, e.g. snr==6
    # average = True or False, if True takes average per subject, False takes epochs (trials, heavy)
    
    dirpathEpochs = datapath + '/myEpochs_' + ActOrPass
    dirpathTFs = datapath + '/myPowers_Active'
    
    for SubID in SubIDs :
        
        print('Start S'+SubID)
        epoData = mne.read_epochs(dirpathEpochs+'/Epoch_'+SubID+'-epo.fif')
        freqs =  np.linspace(1,45,45)
        n_cycles = freqs/2.
        power = mne.time_frequency.tfr_morlet(epoData, freqs=freqs, n_cycles=n_cycles, average = False, use_fft=True, return_itc=False)
        power.apply_baseline(baseline=(-0.5, 0), mode='logratio')
        ff = dirpathTFs +'/Power_' + SubID + '-tfr.h5'
        power.save(ff, overwrite=True)


#%% Run

Powers = calcPower(realSubIDs, 'Active')