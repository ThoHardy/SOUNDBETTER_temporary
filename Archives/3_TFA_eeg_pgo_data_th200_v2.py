# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:14:09 2023

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
pal = [(0.12156862745098039, 0.7058823529411765, 0.4666666666666667),
       (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
       (0.7058823529411765, 0.12156862745098039, 0.12156862745098039)]


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

method = 'morlet' # can be 'morlet' or 'stft'

sf = 1000

freqs = np.linspace(1,30,30)

power_dict = {}
nb_epochs = {i:0 for i in range(10)}
nb_epochs_rejected = {i:0 for i in range(10)}

# Sum up all the epochs
for TaskID in TaskIDs :
    for SubID in SubIDs :
        epochs_this_sub = {i:[] for i in range(10)}
        nb_epochs_this_sub = {i:0 for i in range(10)}
        
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
                
                if abs(peak)<200 or abs(peak)> 400 or start+2.2*sf>len(data[0]) or start-2.2*sf<0:
                    nb_epochs_rejected[score] += 1
                    continue
                    
                epoch = []
                for ch in range(np.shape(data)[0]):
                    epoch.append(data[ch][int(start-2.2*sf):int(start+2.2*sf)])
                    
                # check the eeg amplitude criteria
                max_list = []
                for ch in epoch :
                    normalized_ch = ch - np.mean(ch[int(1.4*sf):int(1.9*sf)])
                    max_list.append(np.max(np.abs(normalized_ch)))
                if max(max_list) > 80:
                    continue
                
                epochs_this_sub[score].append(epoch)
                nb_epochs_this_sub[score] += 1
                
                print('\n... ' + TaskID + SubID + ', run' + str(i) + ' taken...\n')
                
        for score in epochs_this_sub.keys():
            
            if np.shape(epochs_this_sub[score]) == (0,):
                continue
            
            if method == 'morlet' :
                power = mne.time_frequency.tfr_array_morlet(epochs_this_sub[score], sf, freqs, 
                                                        n_cycles=[int(f/2) for f in freqs], 
                                                        zero_mean=True, use_fft=True, 
                                                        decim=1,output='power', n_jobs=1,
                                                        verbose=None)
                
            if method == 'stft' :
                
                power = []
                
                for ep in epochs_this_sub[score] :
                    
                    temp_power = mne.time_frequency.stft(np.array(ep), wsize = 100, 
                                                tstep=None, verbose=None)
                    if power == []:
                        power = temp_power
                    else :
                        power += temp_power
                        
                power = power/len(epochs_this_sub[score])
          
            
            current_mean_power = np.mean(power, axis = 0)
            current_mean_weight = nb_epochs_this_sub[score]
            
            if score in power_dict.keys():
                
                pre_mean_power = power_dict[score]
                pre_weight = nb_epochs[score]
                
                post_mean_power = (pre_weight*pre_mean_power + current_mean_weight*current_mean_power)/(pre_weight+current_mean_weight)
                
                power_dict[score] = post_mean_power
                
            else :
                
                power_dict[score] = current_mean_power
                
            
            nb_epochs[score] += current_mean_weight
            


# Power normalization
for score in power_dict.keys():
    for ch_ind in range(len(power_dict[score])):
        for freq_ind in range(len(power_dict[score][ch_ind])):
            
            power_over_time = power_dict[score][ch_ind][freq_ind]
            
            norm = np.mean(power_over_time[int(1.4*sf):int(1.9*sf)])
            
            power_dict[score][ch_ind][freq_ind] = 10*np.log10(power_over_time/norm)
            


# Plot
for score in [0,5]:
    for central_ch_ind in central_keys:
        plt.figure()
        plt.title('Score ' + str(score) + ', ch. ' + str(central_ch_ind) + ', power for EEG with A(HEOG)>200')
        Z = power_dict[score][central_ch_ind]
        y = freqs
        x = np.linspace(-2.2,2.2,int(4.4*sf))
        X, Y = np.meshgrid(x, y)
        plt.pcolor(X, Y, Z)
        plt.show()
        
        
# Save
for score in power_dict.keys():
    for ch_ind in range(len(power_dict[score])):
        
        future_DF_to_save = {}
        
        for freq_ind in range(len(power_dict[score][ch_ind])):
            
            future_DF_to_save[freqs[freq_ind]] = power_dict[score][ch_ind][freq_ind]
            
        DF_to_save = pandas.DataFrame(future_DF_to_save)
        DF_to_save.to_csv('TFA_S' + str(score) + '_EEG_CH' + str(ch_ind) + 'A(HEOG)_more_than_200.csv')
        



'''

''''''''' '''After reloading the obtained results''' '''''''''''

# Create an MNE TFR structure in order to leverage the MNE plotting tools
mne_power_dict = {}

common_info = mne.create_info(['EEG0' + str(ch_id) for ch_id in central_keys], sf, ch_types = 'eeg')


for score in [0,5]:
    
    powers_this_score = []
    
    for central_ch_ind in central_keys:
        
        powers = pandas.read_csv('TFA_S' + str(score) + '_EEG_CH' + str(central_ch_ind) + 'A(HEOG)_more_than_200.csv')
        
        powers_array = powers.to_numpy().transpose()[1:]
        
        powers_this_score.append(powers_array)
        
        
    TFR_object = mne.time_frequency.AverageTFR(common_info, np.array(powers_this_score), 
                                                   times=np.linspace(-2.2,2.2,int(4.4*sf)), 
                                                   freqs=freqs, nave=nb_epochs[score], 
                                                   comment='REM-locked central EEG data', 
                                                   method='Morlet wavelets, 2 cycles')
    
    for central_ch_ind in central_keys:
        
        TFR_object.plot(picks = ['EEG0' + str(central_ch_ind)], baseline = None, tmin = -2, tmax = 2, dB = True, 
                    colorbar = True, title = 'S' + str(score) + ', EEG0' + str(central_ch_ind))
        
    mne_power_dict[score] = TFR_object
    
    
'''
