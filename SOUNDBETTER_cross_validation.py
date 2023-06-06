# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:02:02 2023

@author: thoma
"""


#%% Import toolboxes and functions, and def 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api
from scipy import stats
from scipy.optimize import minimize 
import seedSOUNDBETTER
import SOUNDBETTER_functions_all_models as all_models
import mne
import os
import scipy
import pandas
from scipy.signal import butter,filtfilt

import warnings
warnings.filterwarnings("ignore") # ignore all warnings

datapath = seedSOUNDBETTER.datapath
os.chdir(datapath)

mat2mne = seedSOUNDBETTER.mat2mne
EmpiricalDistribution = seedSOUNDBETTER.EmpiricalDistribution
CV5_Bifurcation00 = all_models.CV5_Bifurcation00
CV5_Bifurcation01 = all_models.CV5_Bifurcation01
CV5_Unimodal = all_models.CV5_Unimodal00
CV5_Null = all_models.CV5_Null00
CV5_Bayesian01 = all_models.CV5_Bayesian01
CV5_Bayesian00 = all_models.CV5_Bayesian00
CV5_GaussianMixture00 = all_models.Predict_GaussianMixture00
CV5_GaussianMixture00_Sklearn = all_models.Predict_GaussianMixture00_Sklearn
CV5_GaussianMixture00_Sklearn_simulations = all_models.Predict_GaussianMixture00_Sklearn_simulations
CV5_GaussianMixture00_MultiChan = all_models.Predict_GaussianMixture00_MultiChan
Predict_KMeans = all_models.Predict_Kmeans
Predict_KMeans_simulations = all_models.Predict_Kmeans_simulations

realSubIDs = seedSOUNDBETTER.SubIDs

# for conds 0: audibility, 1: blocknumber, 2: evel, 3: vowel, 4: snr, 5: respside

SNRs = np.array([-20,-13,-11,-9,-7,-5])

ind_to_snr = {i+1:SNRs[i] for i in range(6)}

#%% command line
choice = 'kmeans_simulations_bimodal' # can also be : 'multi_chans', 'sklearn', 'time-freq', 'kmeans', 'sklearn_simulations

#%% sklearn on several channels, and raw data

if choice == 'multi_chans':
    list_chans = ['CPz','Pz','P1','P2','POz']
    aud_thresh=[0,1,2,3,4,5] #+ [6,7,8,9]
    list_avg_powers, list_std_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    nb_mixtures = []
    
    for time in np.linspace(100,890,80):
        
        TWOI = [(int(time),int(time+10))] #+ [(350,360)] + [(500,510),(650,660)]
        
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = CV5_GaussianMixture00_MultiChan(list_chans,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI, nb_iterations=1,aud_thresh=aud_thresh,save=False, redo=True)
        
        predictibility_powers, ideal_predictibility_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
        nb_mixtures_this_time = []
        
        for result in cv :
            nb_mixtures_this_time.append(result['best_hyperparams']['n_components'])
            for thresh in aud_thresh : 
                if np.mean(result['means'][0]) < np.mean(result['means'][1]): # if means[0] is mu_low and means[1] is mu_high
                    to_add_actual = np.mean(result['correct_actual_predictions'][thresh])
                else : 
                    to_add_actual = 1-np.mean(result['correct_actual_predictions'][thresh])
                predictibility_powers[thresh].append(to_add_actual)
                ideal_predictibility_powers[thresh].append(np.mean(result['correct_ideal_predictions'][thresh]))
        nb_mixtures.append(np.mean(nb_mixtures_this_time))
        for thresh in aud_thresh :
            list_avg_powers[thresh].append([np.mean(predictibility_powers[thresh]),np.mean(ideal_predictibility_powers[thresh])])
            list_std_powers[thresh].append([np.std(predictibility_powers[thresh])/np.sqrt(20),np.std(ideal_predictibility_powers[thresh])/np.sqrt(20)])
        print('average power : ',{thresh:np.mean(predictibility_powers[thresh]) for thresh in aud_thresh},'\n')
        print('average ideal power : ',{thresh:np.mean(ideal_predictibility_powers[thresh]) for thresh in aud_thresh})
        
    list_avg_powers = {thresh:np.array(list_avg_powers[thresh]).T for thresh in aud_thresh}
    list_sem_powers = {thresh:np.array(list_std_powers[thresh]).T for thresh in aud_thresh}
    
    plt.figure()
    plt.title('Real predictibility with 1D EM on SNR -7, on central electrodes, with Sklearn')
    for thresh in aud_thresh :
        #plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[thresh][1],list_sem_powers[thresh][1],label=str(thresh)+'ideal predictability')
        plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[thresh][0],list_sem_powers[thresh][0],label=str(thresh)+'real_predictibility')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('average number of gaussian mixtures for 1D EM')
    plt.plot(2*np.linspace(100,890,80)-500,nb_mixtures)
    plt.show()

#%% sklearn.GaussianMixture

if choice == 'sklearn':
    aud_thresh=[0,1,2,3,4] #+ [6,7,8,9]
    list_avg_powers, list_sem_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    list_avg_powers_not_heard, list_sem_powers_not_heard = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    list_avg_powers_heard, list_sem_powers_heard = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    list_avg_AUCs, list_std_AUCs = [], []
    list_avg_p_values, list_std_p_values = [], []
    list_avg_consistency, list_std_consistency = [], []
    nb_mixtures = []
    
    times_list =  np.linspace(100,890,20)
    
    for time in times_list :
        
        TWOI = [(int(time),int(time+10))] #+ [(350,360),(500,510),(650,660)]
        
        # remove redondancy
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = CV5_GaussianMixture00_Sklearn(list_SubIDs=list_SubIDs, snr=-9, TWOI=TWOI, nb_iterations=1,aud_thresh=aud_thresh,cv5=True,save=False, redo=True)
        
        predictibility_powers, ideal_predictibility_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
        predictibility_powers_not_heard, predictibility_powers_heard = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
        nb_mixtures_this_time = []
        AUCs_this_time = []
        p_values_this_time = []
        consistencies_this_time = []
        
        for SubID, result in enumerate(cv) :
            nb_mixtures_this_time.append(result['best_hyperparams']['n_components'])
            AUCs_this_time.append(result['AUC_on_audibility'])
            p_values_this_time.append(result['p_value_on_audibility'])
            consistencies_this_time.append(np.mean(result['CV_rand_scores']))
            for thresh in aud_thresh :
                to_add_actual = np.mean(result['correct_actual_predictions'][thresh])
                to_add_actual_not_heard = np.mean(result['correct_actual_predictions_not_heard'][thresh])
                to_add_actual_heard = np.mean(result['correct_actual_predictions_heard'][thresh])
                predictibility_powers[thresh].append(to_add_actual)
                predictibility_powers_not_heard[thresh].append(to_add_actual_not_heard)
                predictibility_powers_heard[thresh].append(to_add_actual_heard)
                ideal_predictibility_powers[thresh].append(np.mean(result['correct_ideal_predictions'][thresh]))
                
            '''
            # save outputs
            os.chdir(datapath+'/Results_Thomas')
            matfile = {'predictibility_powers':predictibility_powers,'predictibility_powers_not_heard':predictibility_powers_not_heard,
                       'predictibility_powers_heard':predictibility_powers_heard,'ideal_predictibility_powers':ideal_predictibility_powers,
                       'AUC':result['AUC']}
            scipy.io.savemat('Results_EM_1DMVPA_500ms_S'+realSubIDs[SubID]+'.mat', matfile)
            os.chdir(datapath)
            '''
        
        nb_mixtures.append(np.mean(nb_mixtures_this_time))
        list_avg_AUCs.append(np.mean(AUCs_this_time))
        list_std_AUCs.append(np.std(AUCs_this_time))
        list_avg_p_values.append(np.nanmean(p_values_this_time))
        list_std_p_values.append(np.nanstd(p_values_this_time))
        list_avg_consistency.append(np.mean(consistencies_this_time))
        list_std_consistency.append(np.std(consistencies_this_time))
        for thresh in aud_thresh :
            list_avg_powers[thresh].append([np.mean(predictibility_powers[thresh]),np.mean(ideal_predictibility_powers[thresh])])
            list_sem_powers[thresh].append([np.std(predictibility_powers[thresh])/np.sqrt(20),np.std(ideal_predictibility_powers[thresh])/np.sqrt(20)])
            list_avg_powers_not_heard[thresh].append(np.mean(predictibility_powers_not_heard[thresh]))
            list_sem_powers_not_heard[thresh].append(np.std(predictibility_powers_not_heard[thresh])/np.sqrt(20))
            list_avg_powers_heard[thresh].append(np.mean(predictibility_powers_heard[thresh]))
            list_sem_powers_heard[thresh].append(np.std(predictibility_powers_heard[thresh])/np.sqrt(20))
        print('average power : ',{thresh:np.mean(predictibility_powers[thresh]) for thresh in aud_thresh},'\n')
        print('average ideal power : ',{thresh:np.mean(ideal_predictibility_powers[thresh]) for thresh in aud_thresh})
        
    list_avg_powers = {thresh:np.array(list_avg_powers[thresh]).T for thresh in aud_thresh}
    list_sem_powers = {thresh:np.array(list_sem_powers[thresh]).T for thresh in aud_thresh}
    
   
    
    plt.figure()
    plt.title('Accuracy of the 1D EM on MVPA data, at SNR -9')
    for thresh in aud_thresh :
        #plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[thresh][1],list_sem_powers[thresh][1],label=str(thresh)+'ideal predictability')
        plt.errorbar(2*times_list-500,list_avg_powers[thresh][0],list_sem_powers[thresh][0],label=str(thresh))
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Precision of the 1D EM on MVPA data, for both heard and not heard trials, at SNR -9, auditory-threshold=3')
    thresh = 3
    plt.errorbar(2*times_list-500,list_avg_powers[thresh][0],list_sem_powers[thresh][0],label='global')
    plt.errorbar(2*times_list-500,list_avg_powers_not_heard[thresh],list_sem_powers_not_heard[thresh],label='not heard')
    plt.errorbar(2*times_list-500,list_avg_powers_heard[thresh],list_sem_powers_heard[thresh],label='heard')
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('AUC of the 1D EM on MVPA data, at SNR -9')
    plt.errorbar(2*times_list-500,list_avg_AUCs,list_std_AUCs)
    plt.xlabel('time')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    
    list_avg_p_values = np.array(list_avg_p_values)
    plt.figure()
    plt.title('p_value of the 1D EM on MVPA data, at SNR -9 (H0 : the means are equal)')
    plt.errorbar(2*times_list-500,list_avg_p_values,list_std_p_values)
    plt.scatter(2*times_list[list_avg_p_values<0.05]-500,[0 for i in range(len(times_list[list_avg_p_values<0.01]))],marker='*')
    plt.xlabel('time')
    plt.ylabel('p_values')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('BIC-deduced average number of gaussian mixtures, for 1D EM on MVPA data, at SNR -9')
    plt.plot(2*times_list-500,nb_mixtures)
    plt.xlabel('time')
    plt.ylabel('avg umber of mixtures')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Average consistency via rand_score on 5CV results, for 1D EM on MVPA data, at SNR -9')
    plt.errorbar(2*times_list-500,list_avg_consistency,np.array(list_std_consistency)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('avg consistency of mixtures')
    plt.legend()
    plt.show()
    
    
#%% sklearn.GaussianMixture on bimodal simulations

if choice == 'sklearn_simulations_bimodal':
    aud_thresh_for_simulation=3
    list_avg_powers, list_sem_powers = [], []
    list_avg_powers_not_heard, list_sem_powers_not_heard = [], []
    list_avg_powers_heard, list_sem_powers_heard = [], []
    list_avg_AUCs, list_std_AUCs = [], []
    list_avg_p_values, list_std_p_values = [], []
    list_avg_consistency, list_std_consistency = [], []
    nb_mixtures = []
    
    times_list =  np.linspace(100,890,20)
    
    for time in times_list :
        
        TWOI = [(int(time),int(time+10))] #+ [(350,360),(500,510),(650,660)]
        
        # remove redondancy
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = CV5_GaussianMixture00_Sklearn_simulations(list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI, nb_iterations=1,bimodal=True,aud_thresh_for_simulation=aud_thresh_for_simulation,cv5=True,save=False, redo=True)
        
        predictibility_powers, ideal_predictibility_powers = [], []
        predictibility_powers_not_heard, predictibility_powers_heard = [], []
        nb_mixtures_this_time = []
        AUCs_this_time = []
        p_values_this_time = []
        consistencies_this_time = []
        
        for SubID, result in enumerate(cv) :
            nb_mixtures_this_time.append(result['best_hyperparams']['n_components'])
            AUCs_this_time.append(result['AUC_on_audibility'])
            p_values_this_time.append(result['p_value_on_audibility'])
            consistencies_this_time.append(np.mean(result['CV_rand_scores']))
            to_add_actual = np.mean(result['correct_actual_predictions'])
            to_add_actual_not_heard = np.mean(result['correct_actual_predictions_not_heard'])
            to_add_actual_heard = np.mean(result['correct_actual_predictions_heard'])
            predictibility_powers.append(to_add_actual)
            predictibility_powers_not_heard.append(to_add_actual_not_heard)
            predictibility_powers_heard.append(to_add_actual_heard)
            ideal_predictibility_powers.append(np.mean(result['correct_ideal_predictions']))
                
            '''
            # save outputs
            os.chdir(datapath+'/Results_Thomas')
            matfile = {'predictibility_powers':predictibility_powers,'predictibility_powers_not_heard':predictibility_powers_not_heard,
                       'predictibility_powers_heard':predictibility_powers_heard,'ideal_predictibility_powers':ideal_predictibility_powers,
                       'AUC':result['AUC']}
            scipy.io.savemat('Results_EM_1DMVPA_500ms_S'+realSubIDs[SubID]+'.mat', matfile)
            os.chdir(datapath)
            '''
        
        nb_mixtures.append(np.mean(nb_mixtures_this_time))
        list_avg_AUCs.append(np.mean(AUCs_this_time))
        list_std_AUCs.append(np.std(AUCs_this_time))
        list_avg_p_values.append(np.nanmean(p_values_this_time))
        list_std_p_values.append(np.nanstd(p_values_this_time))
        list_avg_consistency.append(np.mean(consistencies_this_time))
        list_std_consistency.append(np.std(consistencies_this_time))
        list_avg_powers.append([np.mean(predictibility_powers),np.mean(ideal_predictibility_powers)])
        list_sem_powers.append([np.std(predictibility_powers)/np.sqrt(20),np.std(ideal_predictibility_powers)/np.sqrt(20)])
        list_avg_powers_not_heard.append(np.mean(predictibility_powers_not_heard))
        list_sem_powers_not_heard.append(np.std(predictibility_powers_not_heard)/np.sqrt(20))
        list_avg_powers_heard.append(np.mean(predictibility_powers_heard))
        list_sem_powers_heard.append(np.std(predictibility_powers_heard)/np.sqrt(20))
        print('average power : ',np.mean(predictibility_powers))
        print('average ideal power : ',np.mean(ideal_predictibility_powers)) 
        
    list_avg_powers = np.array(list_avg_powers).T
    list_sem_powers = np.array(list_sem_powers).T
    
   
    
    plt.figure()
    plt.title('Accuracy of the 1D EM on simulated bimodal data, at SNR -7, auditory-threshold=3')
    #plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[thresh][1],list_sem_powers[thresh][1],label=str(thresh)+'ideal predictability')
    plt.errorbar(2*times_list-500,list_avg_powers[0],list_sem_powers[0])
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Precision of the 1D EM on simulated bimodal data, for both heard and not heard trials, at SNR -7, auditory-threshold=3')
    thresh = 3
    plt.errorbar(2*times_list-500,list_avg_powers[0],list_sem_powers[0],label='global')
    plt.errorbar(2*times_list-500,list_avg_powers_not_heard,list_sem_powers_not_heard,label='not heard')
    plt.errorbar(2*times_list-500,list_avg_powers_heard,list_sem_powers_heard,label='heard')
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('AUC of the 1D EM on simulated bimodal data, at SNR -7, auditory-threshold=3')
    plt.errorbar(2*times_list-500,list_avg_AUCs,list_std_AUCs)
    plt.xlabel('time')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    
    list_avg_p_values = np.array(list_avg_p_values)
    plt.figure()
    plt.title('p_value of the 1D EM on simulated bimodal data, at SNR -7 (H0 : the means are equal), auditory-threshold=3')
    plt.errorbar(2*times_list-500,list_avg_p_values,list_std_p_values)
    plt.scatter(2*times_list[list_avg_p_values<0.05]-500,[0 for i in range(len(times_list[list_avg_p_values<0.01]))],marker='*')
    plt.xlabel('time')
    plt.ylabel('p_values')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('BIC-deduced average number of gaussian mixtures, for 1D EM on simulated bimodal data, at SNR -7, auditory-threshold=3')
    plt.plot(2*times_list-500,nb_mixtures)
    plt.xlabel('time')
    plt.ylabel('avg number of mixtures')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Average consistency via rand_score on 5CV results, for 1D EM on simulated bimodal data, at SNR -7, auditory-threshold=3')
    plt.errorbar(2*times_list-500,list_avg_consistency,np.array(list_std_consistency)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('avg consistency of mixtures')
    plt.legend()
    plt.show()
    
#%% sklearn.GaussianMixture on unimodal simulations

if choice == 'sklearn_simulations_unimodal':
    aud_thresh_for_simulation=3
    snr = -7
    list_avg_AUCs, list_std_AUCs = [], []
    list_avg_p_values, list_std_p_values = [], []
    list_avg_consistency, list_std_consistency = [], []
    nb_mixtures = []
    
    times_list =  np.linspace(100,890,20)
    
    for time in times_list :
        
        TWOI = [(int(time),int(time+10))] #+ [(350,360),(500,510),(650,660)]
        
        # remove redondancy
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = CV5_GaussianMixture00_Sklearn_simulations(list_SubIDs=list_SubIDs, snr=snr, TWOI=TWOI, nb_iterations=1,bimodal=False,aud_thresh_for_simulation=aud_thresh_for_simulation,cv5=True,save=False, redo=True)
        
        predictibility_powers, ideal_predictibility_powers = [], []
        predictibility_powers_not_heard, predictibility_powers_heard = [], []
        nb_mixtures_this_time = []
        AUCs_this_time = []
        p_values_this_time = []
        consistencies_this_time = []
        
        for SubID, result in enumerate(cv) :
            nb_mixtures_this_time.append(result['best_hyperparams']['n_components'])
            AUCs_this_time.append(result['AUC_on_audibility'])
            p_values_this_time.append(result['p_value_on_audibility'])
            consistencies_this_time.append(np.mean(result['CV_rand_scores']))
                
            '''
            # save outputs
            os.chdir(datapath+'/Results_Thomas')
            matfile = {'predictibility_powers':predictibility_powers,'predictibility_powers_not_heard':predictibility_powers_not_heard,
                       'predictibility_powers_heard':predictibility_powers_heard,'ideal_predictibility_powers':ideal_predictibility_powers,
                       'AUC':result['AUC']}
            scipy.io.savemat('Results_EM_1DMVPA_500ms_S'+realSubIDs[SubID]+'.mat', matfile)
            os.chdir(datapath)
            '''
        
        nb_mixtures.append(np.mean(nb_mixtures_this_time))
        list_avg_AUCs.append(np.mean(AUCs_this_time))
        list_std_AUCs.append(np.std(AUCs_this_time))
        list_avg_p_values.append(np.nanmean(p_values_this_time))
        list_std_p_values.append(np.nanstd(p_values_this_time))
        list_avg_consistency.append(np.mean(consistencies_this_time))
        list_std_consistency.append(np.std(consistencies_this_time))
    
    
    plt.figure()
    plt.title('AUC of the 1D EM on simulated bimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_AUCs,list_std_AUCs)
    plt.xlabel('time')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    
    list_avg_p_values = np.array(list_avg_p_values)
    plt.figure()
    plt.title('p_value of the 1D EM on simulated bimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_p_values,list_std_p_values)
    plt.scatter(2*times_list[list_avg_p_values<0.05]-500,[0 for i in range(len(times_list[list_avg_p_values<0.01]))],marker='*')
    plt.xlabel('time')
    plt.ylabel('p_values')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('BIC-deduced average number of gaussian mixtures, for 1D EM on simulated bimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.plot(2*times_list-500,nb_mixtures)
    plt.xlabel('time')
    plt.ylabel('avg number of mixtures')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Average consistency via rand_score on 5CV results, for 1D EM on simulated bimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_consistency,np.array(list_std_consistency)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('avg consistency of mixtures')
    plt.legend()
    plt.show()
    
#%% sklearn.KMeans

if choice == 'kmeans':
    aud_thresh=[0,1,2,3,4,5,6,7,8] #+ [6,7,8,9]
    list_avg_powers, list_sem_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    list_avg_powers_not_heard, list_sem_powers_not_heard = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    list_avg_powers_heard, list_sem_powers_heard = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    list_avg_AUCs, list_std_AUCs = [], []
    list_avg_p_values, list_std_p_values = [], []
    list_avg_consistency, list_std_consistency = [], []
    nb_mixtures = []
    
    times_list =  np.linspace(100,890,20)
    
    for time in times_list :
        
        TWOI = [(int(time),int(time+10))] #+ [(350,360),(500,510),(650,660)]
        
        # remove redondancy
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = Predict_KMeans(list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI, nb_iterations=1,aud_thresh=aud_thresh,cv5=True,save=False, redo=True)
        
        predictibility_powers, ideal_predictibility_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
        predictibility_powers_not_heard, predictibility_powers_heard = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
        nb_mixtures_this_time = []
        AUCs_this_time = []
        p_values_this_time = []
        consistencies_this_time = []
        
        for SubID, result in enumerate(cv) :
            AUCs_this_time.append(result['AUC_on_audibility'])
            p_values_this_time.append(result['p_value_on_audibility'])
            consistencies_this_time.append(np.mean(result['CV_rand_scores']))
            for thresh in aud_thresh :
                to_add_actual = np.mean(result['correct_actual_predictions'][thresh])
                to_add_actual_not_heard = np.mean(result['correct_actual_predictions_not_heard'][thresh])
                to_add_actual_heard = np.mean(result['correct_actual_predictions_heard'][thresh])
                predictibility_powers[thresh].append(to_add_actual)
                predictibility_powers_not_heard[thresh].append(to_add_actual_not_heard)
                predictibility_powers_heard[thresh].append(to_add_actual_heard)
                ideal_predictibility_powers[thresh].append(np.mean(result['correct_ideal_predictions'][thresh]))
                
            '''
            # save outputs
            os.chdir(datapath+'/Results_Thomas')
            matfile = {'predictibility_powers':predictibility_powers,'predictibility_powers_not_heard':predictibility_powers_not_heard,
                       'predictibility_powers_heard':predictibility_powers_heard,'ideal_predictibility_powers':ideal_predictibility_powers,
                       'AUC':result['AUC']}
            scipy.io.savemat('Results_EM_1DMVPA_500ms_S'+realSubIDs[SubID]+'.mat', matfile)
            os.chdir(datapath)
            '''
        
        nb_mixtures.append(np.mean(nb_mixtures_this_time))
        list_avg_AUCs.append(np.mean(AUCs_this_time))
        list_std_AUCs.append(np.std(AUCs_this_time))
        list_avg_p_values.append(np.nanmean(p_values_this_time))
        list_std_p_values.append(np.nanstd(p_values_this_time))
        list_avg_consistency.append(np.mean(consistencies_this_time))
        list_std_consistency.append(np.std(consistencies_this_time))
        for thresh in aud_thresh :
            list_avg_powers[thresh].append([np.mean(predictibility_powers[thresh]),np.mean(ideal_predictibility_powers[thresh])])
            list_sem_powers[thresh].append([np.std(predictibility_powers[thresh])/np.sqrt(20),np.std(ideal_predictibility_powers[thresh])/np.sqrt(20)])
            list_avg_powers_not_heard[thresh].append(np.mean(predictibility_powers_not_heard[thresh]))
            list_sem_powers_not_heard[thresh].append(np.std(predictibility_powers_not_heard[thresh])/np.sqrt(20))
            list_avg_powers_heard[thresh].append(np.mean(predictibility_powers_heard[thresh]))
            list_sem_powers_heard[thresh].append(np.std(predictibility_powers_heard[thresh])/np.sqrt(20))
        print('average power : ',{thresh:np.mean(predictibility_powers[thresh]) for thresh in aud_thresh},'\n')
        print('average ideal power : ',{thresh:np.mean(ideal_predictibility_powers[thresh]) for thresh in aud_thresh})
        
    list_avg_powers = {thresh:np.array(list_avg_powers[thresh]).T for thresh in aud_thresh}
    list_sem_powers = {thresh:np.array(list_sem_powers[thresh]).T for thresh in aud_thresh}
    
   
    
    plt.figure()
    plt.title('Accuracy of the 1D kmeans on MVPA data, at SNR -7')
    for thresh in aud_thresh :
        #plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[thresh][1],list_sem_powers[thresh][1],label=str(thresh)+'ideal predictability')
        plt.errorbar(2*times_list-500,list_avg_powers[thresh][0],list_sem_powers[thresh][0],label=str(thresh))
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Precision of the 1D kmeans on MVPA data, for both heard and not heard trials, at SNR -7, auditory-threshold=3')
    thresh = 3
    plt.errorbar(2*times_list-500,list_avg_powers[thresh][0],list_sem_powers[thresh][0],label='global')
    plt.errorbar(2*times_list-500,list_avg_powers_not_heard[thresh],list_sem_powers_not_heard[thresh],label='not heard')
    plt.errorbar(2*times_list-500,list_avg_powers_heard[thresh],list_sem_powers_heard[thresh],label='heard')
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('AUC of the 1D kmeans on MVPA data, at SNR -7')
    plt.errorbar(2*times_list-500,list_avg_AUCs,np.array(list_std_AUCs)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    
    list_avg_p_values = np.array(list_avg_p_values)
    plt.figure()
    plt.title('p_value of the 1D kmeans on MVPA data, at SNR -7 (H0 : the means are equal)')
    plt.errorbar(2*times_list-500,list_avg_p_values,np.array(list_std_p_values)/np.sqrt(20))
    plt.scatter(2*times_list[list_avg_p_values<0.05]-500,[0 for i in range(len(times_list[list_avg_p_values<0.01]))],marker='*')
    plt.xlabel('time')
    plt.ylabel('p_values')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Average consistency via rand_score on 5CV results, for 1D kmeans on MVPA data, at SNR -7')
    plt.errorbar(2*times_list-500,list_avg_consistency,np.array(list_std_consistency)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('avg consistency of mixtures')
    plt.legend()
    plt.show()
    
    
#%% sklearn.KMeans on bimodal simulations

if choice == 'kmeans_simulations_bimodal':
    aud_thresh_for_simulation = 3
    snr = -7
    list_avg_powers, list_sem_powers = [], []
    list_avg_powers_not_heard, list_sem_powers_not_heard = [], []
    list_avg_powers_heard, list_sem_powers_heard = [], []
    list_avg_AUCs, list_std_AUCs = [], []
    list_avg_p_values, list_std_p_values = [], []
    list_avg_consistency, list_std_consistency = [], []
    nb_mixtures = []
    
    times_list =  np.linspace(100,890,20)
    
    for time in times_list :
        
        TWOI = [(int(time),int(time+10))] #+ [(350,360),(500,510),(650,660)]
        
        # remove redondancy
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = Predict_KMeans_simulations(list_SubIDs=list_SubIDs, snr=snr, TWOI=TWOI, nb_iterations=1,aud_thresh_for_simulation=aud_thresh_for_simulation,bimodal=True,cv5=True,save=False, redo=True)
        
        predictibility_powers, ideal_predictibility_powers = [], []
        predictibility_powers_not_heard, predictibility_powers_heard = [], []
        nb_mixtures_this_time = []
        AUCs_this_time = []
        p_values_this_time = []
        consistencies_this_time = []
        
        for SubID, result in enumerate(cv) :
            AUCs_this_time.append(result['AUC_on_audibility'])
            p_values_this_time.append(result['p_value_on_audibility'])
            consistencies_this_time.append(np.mean(result['CV_rand_scores']))
            to_add_actual = np.mean(result['correct_actual_predictions'])
            to_add_actual_not_heard = np.mean(result['correct_actual_predictions_not_heard'])
            to_add_actual_heard = np.mean(result['correct_actual_predictions_heard'])
            predictibility_powers.append(to_add_actual)
            predictibility_powers_not_heard.append(to_add_actual_not_heard)
            predictibility_powers_heard.append(to_add_actual_heard)
            ideal_predictibility_powers.append(np.mean(result['correct_ideal_predictions']))
                
            '''
            # save outputs
            os.chdir(datapath+'/Results_Thomas')
            matfile = {'predictibility_powers':predictibility_powers,'predictibility_powers_not_heard':predictibility_powers_not_heard,
                       'predictibility_powers_heard':predictibility_powers_heard,'ideal_predictibility_powers':ideal_predictibility_powers,
                       'AUC':result['AUC']}
            scipy.io.savemat('Results_EM_1DMVPA_500ms_S'+realSubIDs[SubID]+'.mat', matfile)
            os.chdir(datapath)
            '''
        
        nb_mixtures.append(np.mean(nb_mixtures_this_time))
        list_avg_AUCs.append(np.mean(AUCs_this_time))
        list_std_AUCs.append(np.std(AUCs_this_time))
        list_avg_p_values.append(np.nanmean(p_values_this_time))
        list_std_p_values.append(np.nanstd(p_values_this_time))
        list_avg_consistency.append(np.mean(consistencies_this_time))
        list_std_consistency.append(np.std(consistencies_this_time))
        list_avg_powers.append([np.mean(predictibility_powers),np.mean(ideal_predictibility_powers)])
        list_sem_powers.append([np.std(predictibility_powers)/np.sqrt(20),np.std(ideal_predictibility_powers)/np.sqrt(20)])
        list_avg_powers_not_heard.append(np.mean(predictibility_powers_not_heard))
        list_sem_powers_not_heard.append(np.std(predictibility_powers_not_heard)/np.sqrt(20))
        list_avg_powers_heard.append(np.mean(predictibility_powers_heard))
        list_sem_powers_heard.append(np.std(predictibility_powers_heard)/np.sqrt(20))
        print('average power : ',np.mean(predictibility_powers),'\n')
        print('average ideal power : ',np.mean(ideal_predictibility_powers))
        
    list_avg_powers = np.array(list_avg_powers).T
    list_sem_powers = np.array(list_sem_powers).T
    
   
    
    plt.figure()
    plt.title('Accuracy of the 1D kmeans on simulated bimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_powers[0],list_sem_powers[0])
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Precision of the 1D kmeans on simulated bimodal data, for both heard and not heard trials, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_powers[0],list_sem_powers[0],label='global')
    plt.errorbar(2*times_list-500,list_avg_powers_not_heard,list_sem_powers_not_heard,label='not heard')
    plt.errorbar(2*times_list-500,list_avg_powers_heard,list_sem_powers_heard,label='heard')
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('AUC of the 1D kmeans on simulated bimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_AUCs,np.array(list_std_AUCs)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    
    list_avg_p_values = np.array(list_avg_p_values)
    plt.figure()
    plt.title('p_value of the 1D kmeans on simulated bimodal data (H0 : the means are equal), at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_p_values,np.array(list_std_p_values)/np.sqrt(20))
    plt.scatter(2*times_list[list_avg_p_values<0.05]-500,[0 for i in range(len(times_list[list_avg_p_values<0.01]))],marker='*')
    plt.xlabel('time')
    plt.ylabel('p_values')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Average consistency via rand_score on 5CV results, for 1D kmeans on simulated bimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_consistency,np.array(list_std_consistency)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('avg consistency of mixtures')
    plt.legend()
    plt.show()
    
    
#%% sklearn.KMeans on unimodal simulations

if choice == 'kmeans_simulations_unimodal':
    aud_thresh_for_simulation = 3
    snr = -7
    list_avg_AUCs, list_std_AUCs = [], []
    list_avg_p_values, list_std_p_values = [], []
    list_avg_consistency, list_std_consistency = [], []
    nb_mixtures = []
    
    times_list =  np.linspace(100,890,20)
    
    for time in times_list :
        
        TWOI = [(int(time),int(time+10))] #+ [(350,360),(500,510),(650,660)]
        
        # remove redondancy
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = Predict_KMeans(list_SubIDs=list_SubIDs, snr=snr, TWOI=TWOI, nb_iterations=1,aud_thresh_for_simulation=aud_thresh_for_simulation,bimodal=False,cv5=True,save=False, redo=True)
        
        AUCs_this_time = []
        p_values_this_time = []
        consistencies_this_time = []
        
        for SubID, result in enumerate(cv) :
            AUCs_this_time.append(result['AUC_on_audibility'])
            p_values_this_time.append(result['p_value_on_audibility'])
            consistencies_this_time.append(np.mean(result['CV_rand_scores']))
                
            '''
            # save outputs
            os.chdir(datapath+'/Results_Thomas')
            matfile = {'predictibility_powers':predictibility_powers,'predictibility_powers_not_heard':predictibility_powers_not_heard,
                       'predictibility_powers_heard':predictibility_powers_heard,'ideal_predictibility_powers':ideal_predictibility_powers,
                       'AUC':result['AUC']}
            scipy.io.savemat('Results_EM_1DMVPA_500ms_S'+realSubIDs[SubID]+'.mat', matfile)
            os.chdir(datapath)
            '''
        
        list_avg_AUCs.append(np.mean(AUCs_this_time))
        list_std_AUCs.append(np.std(AUCs_this_time))
        list_avg_p_values.append(np.nanmean(p_values_this_time))
        list_std_p_values.append(np.nanstd(p_values_this_time))
        list_avg_consistency.append(np.mean(consistencies_this_time))
        list_std_consistency.append(np.std(consistencies_this_time))
    
    
    plt.figure()
    plt.title('AUC of the 1D kmeans on simulated unimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_AUCs,np.array(list_std_AUCs)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    
    list_avg_p_values = np.array(list_avg_p_values)
    plt.figure()
    plt.title('p_value of the 1D kmeans on simulated unimodal data (H0 : the means are equal), at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_p_values,np.array(list_std_p_values)/np.sqrt(20))
    plt.scatter(2*times_list[list_avg_p_values<0.05]-500,[0 for i in range(len(times_list[list_avg_p_values<0.01]))],marker='*')
    plt.xlabel('time')
    plt.ylabel('p_values')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Average consistency via rand_score on 5CV results, for 1D kmeans on simulated unimodal data, at SNR '+str(snr)+', auditory_threshold='+str(aud_thresh_for_simulation))
    plt.errorbar(2*times_list-500,list_avg_consistency,np.array(list_std_consistency)/np.sqrt(20))
    plt.xlabel('time')
    plt.ylabel('avg consistency of mixtures')
    plt.legend()
    plt.show()

#%% homemade EM

if choice == 'homemade' :
    aud_thresh=[0,1,2,3,4,5] #+ [4,5,6,7,8,9]
    list_avg_powers, list_std_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
    for time in np.linspace(100,890,80):
        
        TWOI = [(int(time),int(time+5))] + [(350,355),(370,375),(390,395),(410,415),(430,435),(450,455),(470,475),(490,495),(510,515)]
        
        if (time,time+10) in TWOI[1:]:
            TWOI = TWOI[1:]
        
        print('process ', TWOI)
                    
        list_SubIDs=[i for i in range(20)]
        #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
        cv = CV5_GaussianMixture00(list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI, nb_iterations=1,aud_thresh=aud_thresh,save=False, redo=True)
        
        predictibility_powers, ideal_predictibility_powers = {thresh:[] for thresh in aud_thresh}, {thresh:[] for thresh in aud_thresh}
        for result in cv :
            if result == 'NaN':
                continue
            if np.mean(result['full_data_parameters'][1]) < np.mean(result['full_data_parameters'][3]) : #check mu_low < mu_high
                for thresh in aud_thresh : 
                    predictibility_powers[thresh].append(np.mean(result['full_data_correct_prediction'][thresh]))
                    ideal_predictibility_powers[thresh].append(np.mean(result['full_data_ideal_prediction'][thresh]))
            else :
                for thresh in aud_thresh : 
                    predictibility_powers[thresh].append(1-np.mean(result['full_data_correct_prediction'][thresh]))
                    ideal_predictibility_powers[thresh].append(1-np.mean(result['full_data_ideal_prediction'][thresh]))
        for thresh in aud_thresh :
            list_avg_powers[thresh].append([np.mean(predictibility_powers[thresh]),np.mean(ideal_predictibility_powers[thresh])])
            list_std_powers[thresh].append([np.std(predictibility_powers[thresh])/np.sqrt(20),np.std(ideal_predictibility_powers[thresh])/np.sqrt(20)])
        print('average power : ',{thresh:np.mean(predictibility_powers[thresh]) for thresh in aud_thresh},'\n')
        print('average ideal power : ',{thresh:np.mean(ideal_predictibility_powers[thresh]) for thresh in aud_thresh})
        
    list_avg_powers = {thresh:np.array(list_avg_powers[thresh]).T for thresh in aud_thresh}
    list_sem_powers = {thresh:np.array(list_std_powers[thresh]).T for thresh in aud_thresh}
    
    plt.figure()
    plt.title('Real predictibility with 10D EM on SNR -7, on (200,240,280,320,360,400,440,480,520,time), as a function of time')
    for thresh in aud_thresh :
        #plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[thresh][1],list_sem_powers[thresh][1],label=str(thresh)+'ideal predictability')
        plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[thresh][0],list_sem_powers[thresh][0],label=str(thresh)+'real_predictibility')
    plt.legend()
    plt.show()

#%% Conduct CV analysis for the main bifurcation model

#cv = CV5_Bifurcation00(list_SubIDs=list(range(20)),save=True,redo=False,nb_optimizations=5)

#cv = CV5_Unimodal(list_SubIDs=list(range(20)),save=True,redo=False,nb_optimizations=5)

#cv = CV5_Null(list_SubIDs=list(range(20)),save=True,redo=False)

#cv = CV5_Bifurcation01(list_SubIDs=list(range(20)),save=True,redo=False)

#cv = CV5_Bifurcation01(list_SubIDs=list(range(1)),priors=[(0.25,0.5),(-8.4,2),(0.83,2),(2.2,2),(0.06,1)],save=True,redo=False)

#cv = CV5_Bayesian01(list_SubIDs=list(range(7)),save=True,redo=True)

#cv = CV5_Bayesian00(list_SubIDs=list(range(7)),save=True,redo=True)

#cv = all_models.rewrite_CV5_noNoise_Unimodal00(list_SubIDs=list(range(1)), save=True, redo=False)

#cv = all_models.rewrite_CV5_noNoise_Null00(list_SubIDs=list(range(1)), save=True, redo=False)


