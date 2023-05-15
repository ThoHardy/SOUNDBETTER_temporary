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
CV5_GaussianMixture00_Simul = all_models.Predict_GaussianMixture00_SimulatedData

realSubIDs = seedSOUNDBETTER.SubIDs

# for conds 0: audibility, 1: blocknumber, 2: evel, 3: vowel, 4: snr, 5: respside

SNRs = np.array([-20,-13,-11,-9,-7,-5])

ind_to_snr = {i+1:SNRs[i] for i in range(6)}

#%% Conduct CV analysis for the main bifurcation model
'''
list_avg_powers, list_std_powers = [], []
for nb_timepoints in range(1,3):
        
    cv = CV5_GaussianMixture00_Simul([0.8,[-0.2]*nb_timepoints,np.identity(nb_timepoints),[0.2]*nb_timepoints,np.identity(nb_timepoints)])
    
    predictibility_powers, ideal_predictibility_powers = [], []
    for result in cv :
        if result == 'NaN':
            continue
        if np.mean(result['full_data_parameters'][1]) < np.mean(result['full_data_parameters'][3]) :
            predictibility_powers.append(np.mean(result['full_data_correct_prediction']))
            ideal_predictibility_powers.append(np.mean(result['full_data_ideal_prediction']))
        else :
            predictibility_powers.append(1-np.mean(result['full_data_correct_prediction']))
            ideal_predictibility_powers.append(1-np.mean(result['full_data_ideal_prediction']))
        list_avg_powers.append([np.mean(predictibility_powers),np.mean(ideal_predictibility_powers)])
        list_std_powers.append([np.std(predictibility_powers),np.std(ideal_predictibility_powers)])
        print('average power : ',np.mean(predictibility_powers),'\n')
        print('average ideal power : ',np.mean(ideal_predictibility_powers))
        
list_avg_powers = np.array(list_avg_powers).T
list_std_powers = np.array(list_std_powers).T
        
plt.figure()
plt.title('Real and ideal predictibility with nD EM on simulated data, as a function of n')
plt.errorbar(list(range(1,10)),list_avg_powers[1],list_std_powers[1],label='ideal_predictibility')
plt.errorbar(list(range(1,10)),list_avg_powers[0],list_std_powers[0],label='real_predictibility')
plt.legend()
plt.show()

'''
list_avg_powers, list_std_powers = [], []
for time in np.linspace(100,890,80):
    
    TWOI = [(int(time),int(time+10))] #+ [(350,360),(400,410),(450,460)]
    print('process ', TWOI)
                
    list_SubIDs_minus22=[i for i in range(16)]+[i for i in range(17,20)]
    list_SubIDs=[i for i in range(20)]
    #cv = CV5_GaussianMixture00_Simul(bimodal=True,multfactor_std=1,list_SubIDs=list_SubIDs, snr=-7, TWOI=TWOI,save=False, redo=True)
    cv = CV5_GaussianMixture00(list_SubIDs=list_SubIDs, snr=-9, TWOI=TWOI, nb_iterations=1,save=False, redo=True)
    
    predictibility_powers, ideal_predictibility_powers = [], []
    for result in cv :
        if result == 'NaN':
            continue
        if np.mean(result['full_data_parameters'][1]) < np.mean(result['full_data_parameters'][3]) :
            predictibility_powers.append(np.mean(result['full_data_correct_prediction']))
            ideal_predictibility_powers.append(np.mean(result['full_data_ideal_prediction']))
        else :
            predictibility_powers.append(1-np.mean(result['full_data_correct_prediction']))
            ideal_predictibility_powers.append(1-np.mean(result['full_data_ideal_prediction']))
    list_avg_powers.append([np.mean(predictibility_powers),np.mean(ideal_predictibility_powers)])
    list_std_powers.append([np.std(predictibility_powers),np.std(ideal_predictibility_powers)])
    print('average power : ',np.mean(predictibility_powers),'\n')
    print('average ideal power : ',np.mean(ideal_predictibility_powers))
    
list_avg_powers = np.array(list_avg_powers).T
list_std_powers = np.array(list_std_powers).T

plt.figure()
plt.title('Real and ideal predictibility with 4D EM on SNR -9, on (200,300,400,time) as a function of time')
plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[1],list_std_powers[1],label='ideal_predictibility')
plt.errorbar(2*np.linspace(100,890,80)-500,list_avg_powers[0],list_std_powers[0],label='real_predictibility')
plt.legend()
plt.show()

#cv = CV5_Bifurcation00(list_SubIDs=list(range(20)),save=True,redo=False,nb_optimizations=5)

#cv = CV5_Unimodal(list_SubIDs=list(range(20)),save=True,redo=False,nb_optimizations=5)

#cv = CV5_Null(list_SubIDs=list(range(20)),save=True,redo=False)

#cv = CV5_Bifurcation01(list_SubIDs=list(range(20)),save=True,redo=False)

#cv = CV5_Bifurcation01(list_SubIDs=list(range(1)),priors=[(0.25,0.5),(-8.4,2),(0.83,2),(2.2,2),(0.06,1)],save=True,redo=False)

#cv = CV5_Bayesian01(list_SubIDs=list(range(7)),save=True,redo=True)

#cv = CV5_Bayesian00(list_SubIDs=list(range(7)),save=True,redo=True)

#cv = all_models.rewrite_CV5_noNoise_Unimodal00(list_SubIDs=list(range(1)), save=True, redo=False)

#cv = all_models.rewrite_CV5_noNoise_Null00(list_SubIDs=list(range(1)), save=True, redo=False)


