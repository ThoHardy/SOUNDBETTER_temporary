# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:47:21 2023

@author: thoma
"""

import AlphaSOUNDBETTER
import os
import mne
import numpy as np
from scipy.signal import butter,filtfilt


#%% datapath
datapath = AlphaSOUNDBETTER.datapath


#%% subjects' ids
SubIDs = ['01','02','03','05','06','07','08','09','11','12','13','14','15','17','19','20','22','23','24','25']


#%% mat2mne function

def mat2mne(file_name,active):
    # converts the matlab fieldtrip file with name file_name to mne format
    # active: is it an active or passive session (true or false)
    
    import scipy.io as sio
    from mne.epochs import EpochsArray
    from pandas import DataFrame
    
    # Read channel struture from mne example
    raw_fname = os.path.join(datapath, 'template_raw.fif')
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
    epochs.metadata = y
    return epochs



#%% Test the function and plot the corresponding evoked data
def plot_mat2mne_evokeds():
    plot = True

    if plot :
    
        epochs = mat2mne(os.path.join(datapath,'Subject_01_Active','data_ref.mat'),1)
        evoked = epochs.average(picks=None,method='mean',by_event_type=True)
    
        # plot
        for ev in evoked : 
            ev.pick('Cz').plot(titles='SNR '+ev.comment)



#%% Create an array with the (snr*projcted activity) distributions for one subject

def EmpiricalDistribution(times,preds,conds,blocks=np.linspace(1,20,20)):
    
    '''
    Parameters
    ----------
    times : tuple (t_min,t_max) of the time interval where to average the activity data (in seconds).
    preds : list or array obtained by classif['preds'][0][SubID].
    conds : list or array obtained by classif['conds'][0][SubID].
    blocks : list or array of int being the blocks to use. Optional (default : all blocks).

    Returns
    -------
    Dictionary with SNRs as keys and the corresponding average activity lists as values.

    '''
    
    # constants
    SNRs = np.array([-20,-13,-11,-9,-7,-5])
    ind_to_snr = {i+1:SNRs[i] for i in range(6)}
    sfreq = 500
    
    # unpack the input
    t_min, t_max = times
    snr_array = conds[:,4]
    block_array = conds[:,1]
    
    # initialize future output
    empirical_distributions = {x:[] for x in SNRs}
    
    # lowpass filter the data
    def butter_lowpass_filter(data, cutoff, fs, order):
        nyq = 0.5*fs
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return(y)
    preds_filtered = butter_lowpass_filter(preds, cutoff=10, fs=sfreq, order=12)
    
    # fill in the empirical distributions dict
    for epo_ind, epo in enumerate(preds_filtered):
        
        snr = ind_to_snr[snr_array[epo_ind]]
        block = block_array[epo_ind]
        
        # select just the blocks of interest
        if not(block in blocks):
            continue
        
        # post-onset distribution
        average_value = np.mean(epo[int(t_min*sfreq):int(t_max*sfreq)])
        empirical_distributions[snr].append(average_value)
        
    return(empirical_distributions)
