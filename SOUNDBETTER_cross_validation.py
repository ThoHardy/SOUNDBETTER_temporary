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
fit_Bayesian00 = all_models.fit_Bayesian00
test_Bayesian00 = all_models.test_Bayesian00

realSubIDs = seedSOUNDBETTER.SubIDs

file_name = os.path.join('Subject_All_Active',
                         'classif_SingleTrialPred_vowelpresence_train_maxSNR_test_allSNR_active_20_subjects')
classif = scipy.io.loadmat(file_name)
preds, conds = classif['preds'][0], classif['conds'][0]
# for conds 0: audibility, 1: blocknumber, 2: evel, 3: vowel, 4: snr, 5: respside

SNRs = np.array([-20,-13,-11,-9,-7,-5])

ind_to_snr = {i+1:SNRs[i] for i in range(6)}

#%% Try functions

times = (1-0.016,1+0.014)
preds, conds = preds[1], conds[1]

data_train = EmpiricalDistribution(times,preds,conds,blocks=np.linspace(1,16,16))

# compute initial parameters
mu_noise_guess, std_noise_guess = np.mean(data_train[-20]), np.std(data_train[-20])
L_high_guess = np.mean(data_train[-5]) - np.mean(data_train[-13])
k_guess, k_high_guess = L_high_guess/8, L_high_guess/8
step_guess = np.mean(data_train[-13]) - np.mean(data_train[-20])
threshsnr_guess = -9

iparameters = np.array([k_guess, threshsnr_guess, step_guess, L_high_guess, 
                        k_high_guess, mu_noise_guess, std_noise_guess])
priors = [None, None, None, None, None, None, None,]

success, parameters, fun, nb_points = fit_Bayesian00(data_train, iparameters, priors)

print(success, parameters, fun, nb_points)
print('\n average likelihood on training set : ', np.exp(-fun/nb_points))

data_test = EmpiricalDistribution(times,preds,conds,blocks=np.linspace(17,20,4))

avg_lh = test_Bayesian00(data_test,parameters)

print('\n avergae likelihood for testing set : ', avg_lh )