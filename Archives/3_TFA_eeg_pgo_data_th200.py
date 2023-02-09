# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:59:45 2023

@author: thoma
"""


import os
import numpy as np
import mne
import AlphaREM2PGO
import scipy.io
import pandas
import matplotlib.pyplot as plt
#import ptitprince as pp
import seaborn as sns
sns.set(font_scale=1.2)
import math

os.chdir(AlphaREM2PGO.path_data+'mne_analysis')

FullTaskIDs = ['M1_AAAAB_Arithm','M2_ABABA_Arithm','M3_ABABA_ABCD']
TaskIDs = ['M1','M2','M3']
SubIDs = ['s0'+str(i) for i in range(1,10)] + ['s10','s11'] + ['s'+str(i) for i in range(14,30)]
StageIDs = ['Wake','N1','N2','N3','N4','REM']
stage_dict = {'Wake':1,'N1':2,'N2':3,'N3':4,'N4':5, 'REM':6}

# colour palette for Wake, NREM, and REM
pal = [(0.12156862745098039, 0.7058823529411765, 0.4666666666666667),(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),(0.7058823529411765, 0.12156862745098039, 0.12156862745098039)]


'''''''' '''Central EEG Channels''' ''''''''''''
central_names = AlphaREM2PGO.central_names
central_colors = AlphaREM2PGO.central_colors
central_keys = AlphaREM2PGO.central_keys
''''''''''''''''''

'''''''' '''Epochs Selection Criteria''' ''''''''

selection_criteria = AlphaREM2PGO.selection_criteria

# Plus two implicit criteria : the peak is supposed to be less than 1s after the onset ; 
# and amp1*amp2<0 (phase opposition).
''''''''''''''

'''''''' '''Import Project Basic Functions''' ''''''''
select_movement = AlphaREM2PGO.select_movement
''''''''''''''

sf = 1000

channels = ['EEG0' + str(k) for k in central_keys]

epochs_dict = {i:[] for i in range(10)}
nb_epochs = {i:0 for i in range(10)}
nb_epochs_rejected = {i:0 for i in range(10)}

# Sum up all the epochs
for TaskID in TaskIDs :
    for SubID in SubIDs :
        for i in range(1,20):
            
            try :
                os.chdir(AlphaREM2PGO.path_data+'mne_analysis')
                raw = mne.io.read_raw_fif(TaskID + SubID + '_filtered_restart_raw-' + str(i) + '.fif',preload=True).apply_proj()
                os.chdir(AlphaREM2PGO.path_data + 'Preproc')
                metadata = pandas.read_csv(TaskID + '_' + SubID + '_' + str(i) + '_' + 'metadata_detect_REMs.csv')
                os.chdir(AlphaREM2PGO.path_data+'mne_analysis')
            except FileNotFoundError:
                continue
            
            data = raw.copy().pick_types(eeg=True).get_data()  
        
            for j in range(len(metadata)):
                
                peak = metadata['Peak_Broadband_ampl'][j]
                start = metadata['Starts'][j]
                score = metadata['Score'][j]
                
                if abs(peak)<200 or start+2.2*sf>len(data[0]) or start-2.2*sf<0:
                    nb_epochs_rejected[score] += 1
                    continue
                    
                epoch = data[:][int(start-2.2*sf):int(start+2.2*sf)]
                
                epochs_dict[score].append(epoch)
                nb_epochs[score] += 1
                
                print('\n... ' + TaskID + SubID + ', run' + str(i) + ' taken...\n')
                
            

# Create an mne.Epochs object

raw = mne.io.read_raw_fif('M1s01_filtered_restart_raw-3.fif').pick_types(eeg=True) # just for the metadata
raw_info =  raw.info()
ch_names = raw_info['ch_names']
sfreq = sf
ch_types = 'eeg'

info = mne.create_info(ch_names, sfreq, ch_types, verbose=None)

for score in epochs_dict.keys():
    
    epochs = mne.EpochsArray(epochs_dict[score], info, events=None, tmin=2.2, event_id=None, 
                         reject=None, flat=None, reject_tmin=None, reject_tmax=None, 
                         baseline=(-2,-1), proj=True, on_missing='raise', metadata=None)
    
    freqs = np.linspace(1,30,30)
    
    power, itc = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=2, use_fft=True, return_itc=True, 
                                  decim=1, n_jobs=1, picks=None, zero_mean=True, 
                                  average=True, output='power', verbose=None)
    
    plt.figure()
    plt.title('Score ' + str(score) + ' power for EEG with A(HEOG)>200')
    power.plot()
    plt.show()
    
    plt.figure()
    plt.title('Score ' + str(score) + ' ITC for EEG with A(HEOG)>200')
    power.plot()
    plt.show()



