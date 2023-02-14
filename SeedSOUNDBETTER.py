# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:12:31 2023

@author: thoma
"""

import AlphaSOUNDBETTER
import os
import mne
import numpy as np


#%% datapath
datapath = AlphaSOUNDBETTER.datapath


#%% subjects' ids
SubIDs = ['01']


#%% mat2mne function

def mat2mne(file_name,active):
    # converts the matlab fieldtrip file with name file_name to mne format
    # active: is it an active or passive session (true or false)
    
    import scipy.io as sio
    from mne.epochs import EpochsArray
    from pandas import DataFrame
    
    # Read channel struture from mne example
    raw_fname = datapath + 'template_raw.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.pick_types(eeg=True) # But there is already just eeg channels ?
    
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
                   np.zeros(n_trial), np.array(y['snr'])].astype(int)
    epochs = EpochsArray(data, info, events=events, tmin=ft_data.time[0][0],
                         verbose=False)
    return epochs, y



#%% Test the function and plot the corresponding evoked data
 
epochs, y = mat2mne(datapath + os.path.join('Subject_01_Active','data_ref.mat'),1)
evoked = epochs.average(picks=None,method='mean',by_event_type=True)
 
 # plot
for ev in evoked : 
    ev.pick('Cz').plot(titles='SNR '+ev.comment)


#%%
