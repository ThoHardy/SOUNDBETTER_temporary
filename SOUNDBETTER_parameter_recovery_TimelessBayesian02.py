# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:38:03 2023

@author: thoma
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt


maxiter = 800

def EM_parameter_recovery(nb_timepoints,plot = False, print_steps = False):

    # 1st step : stimulation of the data
    
    # parameters
    
    nb_trials_per_timepoint = 130
    
    ratio_list = np.linspace(0.2,5,60)
    
    real_mu_high, real_mu_low = 1, -1
    while real_mu_high < real_mu_low:
        real_mu_high, real_mu_low = np.random.normal(1,1), np.random.normal(-1,1)
    real_sigma_list = (real_mu_high - real_mu_low)/ratio_list
    real_beta = 0.65
    
    # data stimulation
    
    data_list = []
    labels_list = []
    nb_high = int(nb_trials_per_timepoint*real_beta)
    
    for sigma in real_sigma_list :
        data_this_sigma = []
        labels_this_sigma = []
        for trial in range(nb_trials_per_timepoint):
            if trial < nb_high :
                data_this_sigma.append(np.random.normal(real_mu_high,sigma,nb_timepoints))
                labels_this_sigma.append(1)
            else : 
                data_this_sigma.append(np.random.normal(real_mu_low,sigma,nb_timepoints))
                labels_this_sigma.append(0)
        rand_inds = np.random.permutation([i for i in range(len(data_this_sigma))])
        data_list.append(np.array(data_this_sigma)[rand_inds])
        labels_list.append(np.array(labels_this_sigma)[rand_inds])
    
        
    # 2nd step : parameter recovery
    
    convergence_list = []
    mu_high_list, mu_low_list = [], []
    sigma_list = []
    beta_list = []
    LLs_list = []
    HSP_list_list = []
    
    mu_high_guess, mu_low_guess = np.random.normal(1,1,nb_timepoints), np.random.normal(-1,1,nb_timepoints)
    
    for ind_data, data in enumerate(data_list) : # data is one set of 130 points, corresponding to one dmu/std ratio
        
        if ind_data%20 == 0 and print_steps:
            print('\n processing data nb ', ind_data)
        
        mu_high, mu_low = [mu_high_guess], [mu_low_guess]
        sigma = [np.identity(nb_timepoints)]
        beta = [0.5]
        
        # compute probabilities of belonging to the higher state based on guessed parameters
        HSP_list = []
        for x in data :
            try :
                a, b = multivariate_normal.pdf(x,mu_high[-1],sigma[-1]), multivariate_normal.pdf(x,mu_low[-1],sigma[-1])
            except : 
                a, b = multivariate_normal.pdf(x,mu_high[-1],sigma[-1],allow_singular=True), multivariate_normal.pdf(x,mu_low[-1],sigma[-1],allow_singular=True)
            HSP_list.append(beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
        
        # iterate over the parameters' versions
        LLs = []
        convergence = 0
        i = 0
        while i<maxiter and convergence < 10 :
                
            # update the parameters
            beta.append(np.mean(HSP_list))
            
            # usual way to iterate the mus
            mu_high.append(np.nansum([HSP_list[k]*data[k] for k in range(len(data))],0)/np.nansum(HSP_list))
            mu_low.append(np.nansum([(1-HSP_list[k])*data[k] for k in range(len(data))],0)/np.sum([1-hsp for hsp in HSP_list]))
            
            '''
            # k-means inspired way to iterate the mus
            predicted_high, predicted_low = [], []
            for ind_x, x in enumerate(data) :
                if HSP_list[ind_x] > 0.5:
                    predicted_high.append(x)
                else :
                    predicted_low.append(x)
            mu_high.append(np.mean(predicted_high,0))
            mu_low.append(np.mean(predicted_low,0))
            '''
            
            sigma_high_temporary = np.nansum([HSP_list[k]*np.matmul(np.transpose(np.array([data[k]-mu_high[-1]])),np.array([data[k]-mu_high[-1]])) for k in range(len(data))],0)/np.nansum(HSP_list)
            sigma_low_temporary = np.nansum([(1-HSP_list[k])*np.matmul(np.transpose(np.array([data[k]-mu_low[-1]])),np.array([data[k]-mu_low[-1]])) for k in range(len(data))],0)/np.nansum([1-hsp for hsp in HSP_list])
            sigma.append((sigma_high_temporary + sigma_low_temporary)/2) # average of the supposed stds for the two distributions
            
            # update HSP_list 
            for ind_x, x in enumerate(data) :
                try :
                    a, b = multivariate_normal.pdf(x,mu_high[-1],sigma[-1]), multivariate_normal.pdf(x,mu_low[-1],sigma[-1])
                except : 
                    a, b = multivariate_normal.pdf(x,mu_high[-1],sigma[-1],allow_singular=True), multivariate_normal.pdf(x,mu_low[-1],sigma[-1],allow_singular=True)
                HSP_list[ind_x] = (beta[-1]*a/(beta[-1]*a + (1-beta[-1])*b))
                
            # compute the obtained log-likelihood
            new_LL = 0
            try :
                inv_sigma = np.linalg.inv(sigma[-1])
            except :
                inv_sigma = np.linalg.pinv(sigma[-1])
            for ind_x, x in enumerate(data) :
                if HSP_list[ind_x] >= 0.5 :
                    new_LL -= np.log(beta[-1]) - 0.5*np.log(np.linalg.det(sigma[-1])) - 0.5*np.matmul(np.transpose(x-mu_high[-1]),np.matmul(inv_sigma,x-mu_high[-1])) - (nb_timepoints/2)*np.log(2*np.pi)
                else :
                    new_LL -= np.log(1-beta[-1]) - 0.5*np.log(np.linalg.det(sigma[-1])) - 0.5*np.matmul(np.transpose(x-mu_low[-1]),np.matmul(inv_sigma,x-mu_low[-1])) - (nb_timepoints/2)*np.log(2*np.pi)
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
        HSP_list_list.append(HSP_list)
    
    
        
    new_mu_high_list, new_mu_low_list = [], []
    for k in range(len(mu_high_list)):
        if np.mean(mu_high_list[k]) > np.mean(mu_low_list[k]):
            new_mu_high_list.append(mu_high_list[k])
            new_mu_low_list.append(mu_low_list[k])
        else :
            new_mu_high_list.append(mu_low_list[k])
            new_mu_low_list.append(mu_high_list[k])
                
    
        
    plt.figure()
        
    plt.subplot(2,2,1)
    plt.title('Estimations of mu_high against ratio')
    for ind_dim in range(nb_timepoints) :
        plt.plot(ratio_list, [new_mu_high_list[k][ind_dim] for k in range(len(mu_high_list))],label='mu_high',color='red')
        plt.plot(ratio_list, [new_mu_low_list[k][ind_dim] for k in range(len(mu_low_list))],label='mu_low',color='blue')
    
    
    # plot if dimension >= 2, the first two axis
    if nb_timepoints >= 2:
        
        ind_ratio = 19
    
        data = data_list[ind_ratio]
        HSP_list = HSP_list_list[ind_ratio]
        
        predicted_high, predicted_low = [], []
        for ind_x, x in enumerate(data) :
            if HSP_list[ind_x] > 0.5:
                predicted_high.append(x)
            else :
                predicted_low.append(x)
        
        plt.subplot(2,2,2)
        plt.scatter([predicted_high[k][0] for k in range(len(predicted_high))],[predicted_high[k][1] for k in range(len(predicted_high))],color='red')
        plt.scatter([predicted_low[k][0] for k in range(len(predicted_low))],[predicted_low[k][1] for k in range(len(predicted_low))],color='blue')
        plt.scatter(new_mu_high_list[-1][0],new_mu_high_list[-1][1],color='grey',marker='^')
        plt.scatter(new_mu_low_list[-1][0],new_mu_low_list[-1][1],color='grey',marker='v')
        plt.title('dmu/std ratio : ' + str(ratio_list[ind_ratio]))
        
    
    # plot the ratio of clustering success for each dmu/std ratio
    performance_list = []
    for ind_data, data in enumerate(data_list):
        
        if ind_data < 4:
            continue
        
        HSP_list = HSP_list_list[ind_data]
        labels = labels_list[ind_data]
        mu_high = new_mu_high_list[ind_data]
        mu_low = new_mu_low_list[ind_data]
        nb_successes = 0
        
        predicted_high, predicted_low = [], []
        for ind_x, x in enumerate(data) :
            if HSP_list[ind_x] > 0.5:
                predicted_high.append(x)
            else :
                predicted_low.append(x)
        
        if np.matmul(mu_high,np.mean(predicted_high,0)) > np.matmul(mu_low,np.mean(predicted_high,0)):
            for ind_x, x in enumerate(data):
                if (HSP_list[ind_x]>=0.5 and labels[ind_x] == 1) or (HSP_list[ind_x]<0.5 and labels[ind_x] == 0):
                    nb_successes += 1
        else : 
            for ind_x, x in enumerate(data):
                if (HSP_list[ind_x]<0.5 and labels[ind_x] == 1) or (HSP_list[ind_x]>=0.5 and labels[ind_x] == 0):
                    nb_successes += 1
            
            
        performance_list.append(nb_successes/len(data))
    
    plt.subplot(2,1,2)
    plt.plot(ratio_list[4:],performance_list)
    if nb_timepoints >= 2:
        plt.plot([ratio_list[ind_ratio],ratio_list[ind_ratio]],[0,1],linestyle='--')
    
    if plot :
        plt.show()
    else :
        plt.close()
    
    return(convergence_list, new_mu_high_list, new_mu_low_list, sigma_list, beta_list, LLs_list, performance_list)



# Plot the (dimension) * (performance) graph
# ratio_list[8] = 0.76
# ratio_list[5] = 0.52
# ratio_list[4] = 0.43

performance_per_dim = []
std_per_dim = []
nb_max_dim = 5
nb_opti = 3

for k in range(1,nb_max_dim+1):
    print('Covering dimension',k)
    perf_this_dim = []
    nb_trial = 0
    while nb_trial < nb_opti:
        print('nb_trial :', k, ', ', nb_trial)
        try :
            convergence_list, new_mu_high_list, new_mu_low_list, sigma_list, beta_list, LLs_list, performance_list = EM_parameter_recovery(nb_timepoints=k,plot = False)
        except ValueError :
            continue
        perf_this_dim.append(performance_list[4])
        nb_trial += 1
    performance_per_dim.append(np.mean(perf_this_dim))
    std_per_dim.append(np.std(perf_this_dim))
    
    
plt.figure()
plt.title('average of 30 optimizations for each nb of dimension')
plt.errorbar(list(range(1,nb_max_dim+1)),performance_per_dim,yerr=std_per_dim)
plt.savefig('ClassifPerfEM_v3.png')
plt.show()


