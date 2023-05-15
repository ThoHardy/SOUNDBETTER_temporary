# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:55:21 2023

@author: thoma
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


maxiter = 800


#%% 1st step : stimulation of the data

# parameters

nb_trials = 130

ratio_list = np.linspace(0.1,10,100)

real_mu_high, real_mu_low = 1, -1
while real_mu_high < real_mu_low:
    real_mu_high, real_mu_low = np.random.normal(1,1), np.random.normal(-1,1)
real_sigma_list = (real_mu_high - real_mu_low)/ratio_list
real_beta = 0.65

# data stimulation

data_list = []
nb_high = int(nb_trials*real_beta)

for sigma in real_sigma_list :
    data_this_sigma = []
    for trial in range(nb_trials):
        if trial < nb_high :
            data_this_sigma.append(np.random.normal(real_mu_high,sigma))
        else : 
            data_this_sigma.append(np.random.normal(real_mu_low,sigma))
    data_list.append(np.random.permutation(data_this_sigma))
    
#%% 2nd step : parameter recovery

convergence_list = []
mu_high_list, mu_low_list = [], []
sigma_list = []
beta_list = []
LLs_list = []

mu_high_guess, mu_low_guess = np.random.normal(1,1), np.random.normal(-1,1)
while mu_high_guess < mu_low_guess:
    mu_high_guess, mu_low_guess = np.random.normal(1,1), np.random.normal(-1,1)

for ind_data, data in enumerate(data_list) :
    
    if ind_data%20 == 0:
        print('\n processing data nb ', ind_data)
    
    mu_high, mu_low = [mu_high_guess], [mu_low_guess]
    sigma = [1]
    beta = [0.5]
    
    # compute probabilities of belonging to the higher state based on guessed parameters
    HSP_list = []
    for x in data :
        a, b = norm.pdf(x,mu_high[-1],sigma[-1]), norm.pdf(x,mu_low[-1],sigma[-1])
        HSP_list.append(beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
    
    # iterate over the parameters' versions
    LLs = []
    convergence = 0
    i = 0
    while i<maxiter and convergence < 10 :
            
        # update the parameters
        beta.append(np.mean(HSP_list))
        mu_high.append(np.sum([HSP_list[k]*data[k] for k in range(len(data))])/np.sum(HSP_list))
        sigma_high_temporary = np.sum([HSP_list[k]*(data[k]-mu_high[-1])**2 for k in range(len(data))])/np.sum(HSP_list)
        sigma_low_temporary = np.sum([(1-HSP_list[k])*(data[k]-mu_low[-1])**2 for k in range(len(data))])/np.sum([1-hsp for hsp in HSP_list])
        sigma.append((sigma_high_temporary + sigma_low_temporary)/2) # average of the supposed stds for the two distributions
        mu_low.append(np.sum([(1-HSP_list[k])*data[k] for k in range(len(data))])/np.sum([1-hsp for hsp in HSP_list]))
        
        # update HSP_list      
        for ind_x, x in enumerate(data) :
            a, b = norm.pdf(x,mu_high[-1],sigma[-1]), norm.pdf(x,mu_low[-1],sigma[-1])
            HSP_list[ind_x] = (beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
            
        # compute the obtained log-likelihood
        new_LL = 0
        for ind_x, x in enumerate(data) :
            if HSP_list[ind_x] >= 0.5 :
                new_LL -= np.log(beta[-1]) - 0.5*np.log(sigma[-1]) - 0.5*((x-mu_high[-1])**2)/sigma[-1] - 0.5*np.log(2*np.pi)
            else :
                new_LL -= np.log(1-beta[-1]) - 0.5*np.log(sigma[-1]) - 0.5*((x-mu_low[-1])**2)/sigma[-1] - 0.5*np.log(2*np.pi)
        LLs.append(new_LL)
        
        # assess convergence
        if len(LLs) > 1 :
            if LLs[-2] - LLs[-1] < 1:
                convergence += 1
            else :
                convergence = 0
            
        i += 1

    convergence_list.append(convergence)
    mu_high_list.append(mu_high[-1])
    mu_low_list.append(mu_low[-1])
    sigma_list.append(sigma[-1])
    beta_list.append(beta[-1])
    LLs_list.append(LLs[-1])

'''
plt.figure()
plt.title('LLH as a function of the ratio')
plt.plot(ratio_list, -np.array(LLs_list))
plt.show()
'''
plt.figure()
plt.subplot(1,2,1)
plt.title('Estimations of sigma and real sigma against ratio')
plt.plot(ratio_list, sigma_list,label='estimated')
plt.plot(ratio_list, real_sigma_list,linestyle='--',label='real')
plt.legend()
plt.subplot(1,2,2)
plt.title('Estimations of mu_high/low and real mu_high/low against ratio')
plt.plot(ratio_list, mu_high_list,color='blue',label='mu_high')
plt.plot(ratio_list, [real_mu_high for i in range(len(ratio_list))],color='blue',linestyle='--')
plt.plot(ratio_list, mu_low_list,color='orange',label='mu_low')
plt.plot(ratio_list, [real_mu_low for i in range(len(ratio_list))],color='orange',linestyle='--')
plt.show()