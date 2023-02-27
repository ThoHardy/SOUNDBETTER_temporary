# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:31:11 2023

@author: thoma
"""

#%% Compare the results of the different models. 

# 1st : amount of convergence success per model. 
# 2dn : average likelihood per model.
# 3rd : compare (mu_noise,std_noise) for _noPrior and _withPrior
# 4th : plot the estimated threshsnr to see if it is indeed around -9.
# The "rightness of fitting" is quantified with these two plots. 
# Then, in order to evaluate the models qualitatively :
# 5th : compare (cloudplot) the values of k_high for the bounded and the free model,
# to see if the free model tends toward an affine regression.
# 6th : compare (cloudplot) the values of k for the bounded and the free models.

#%% Description of the models :
    
    # _noPrior : all the 7 parameters are fitted at once via MLE
    # _withPrior00 : mu_noise and std_noise are pre_fitted on trials with no sound
    # _withPrior01 : _withPrior00 + absolute values of k_high and k used in the equations.
    # _withPrior02 : _withPrior01 + absolute value of L_high used in the equation.
    # _withPrior03 : _withPrior01 + bound of L_high between 0 and L_high_guess + std_noise.
    # _withPrior04 : _withPrior01 + (step = mu_noise)
    # _withPrior05 : _withPrior00 + threshsnr is fixed at -9dB.
    
    # _noPrior_ReLu : y=ReLu(ax+b). All the 6 parameters are fitted at once via MLE
    # _withPrior_ReLu : mu_noise and std_noise are pre_fitted on trials with no sound.
    
    # _noPrior_2DPoly : 2nd degree polynomial fit (3 free parameters + 2 noise parameters)
    # _withPrior_2DPoly00 : mu_noise and std_noise are pre_fitted on trials with no sound
    # _withPrior_2DPoly01 : _withPrior_2DPoly00 + threshsnr is fixed at -9dB.
    
    # _noPrior_3DPoly : 3rd degree polynomial fit (4 free parameters + 2 noise parameters)
    # _withPrior_3DPoly00 : mu_noise and std_noise are pre_fitted on trials with no sound
    # _withPrior_3DPoly01 : _withPrior_3DPoly00 + threshsnr is fixed at -9dB.

#%% Compare the results of the different models.

import seedSOUNDBETTER
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model

os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Model_Fit')

SubIDs = seedSOUNDBETTER.SubIDs

model_ids = ['_noPrior','_withPrior00',
             '_fixedBoundaries',
             #'_noPrior_ReLu',
             '_noPrior_1DPoly',
             '_noPrior_2DPoly','_noPrior_2DPoly_concave',
             '_noPrior_3DPoly']

bayesian_model_ids = ['_Bayesian00','_Bayesian_2DPoly']

convergence_scores = {imodel:0 for imodel in model_ids}
average_likelihood_lists = {imodel:[] for imodel in model_ids}
noise_parameters_lists = {imodel:[] for imodel in model_ids}
threshsnrs_lists = [[],[]]
k_high_lists = [[],[]]

for ind_model, imodel in enumerate(model_ids) :
    
    for SubID in SubIDs :
        
        df = pandas.read_csv('ModelFit_Subject'+SubID+imodel)
        
        # 1st (convergence)
        if df['success'][0]:
            convergence_scores[imodel] += 1
            
        # 2nd (avg likelihood)
        average_likelihood_lists[imodel].append(df['average_likelihood'][0])
        
        # 3rd (noise parameters)
        noise_parameters_lists[imodel].append([df['mu_noise'][0],df['std_noise'][0]])
        
        # 4th (threshsnr)
        if imodel != '_withPrior_3DPoly01' :
            threshsnrs_lists[0].append(imodel)
            threshsnrs_lists[1].append(df['threshsnr'][0])
       
        
        # 5th (k_high) 
        if ind_model < 2 :
            k_high_lists[0].append(imodel)
            k_high_lists[1].append(df['k_high'][0])

# 1st (convergence)
convergence_ratios = {imodel : convergence_scores[imodel]/len(SubIDs) for imodel in model_ids}
plt.figure()
plt.title('Convergence ratio for each model')
plt.bar(range(len(model_ids)), list(convergence_ratios.values()), tick_label=list(convergence_ratios.keys()))
plt.show()
    
# 2nd (avg likelihood)
average_likelihood = {imodel : np.mean(average_likelihood_lists[imodel]) for imodel in average_likelihood_lists.keys()}
plt.figure()
plt.title('Relative variations of average likelihood per model')
mean_value = np.mean(list(average_likelihood.values()))
relative_values = [(list(average_likelihood.values())[i] - mean_value)/mean_value for i in range(len(model_ids))]
plt.bar(range(len(model_ids)), relative_values, tick_label=list(model_ids))
plt.show()

# 3rd (noise parameters)
noise_parameters_lists = {key: np.array(noise_parameters_lists[key]) for key in noise_parameters_lists.keys()}
df = {'x':noise_parameters_lists['_noPrior'][:,0], 'y':noise_parameters_lists['_withPrior00'][:,0]}
regr = linear_model.LinearRegression()
regr.fit(np.array([[elmt] for elmt in df['x']]),np.array([[elmt] for elmt in df['y']]))
a,b,r = regr.coef_, regr.intercept_, regr._residues
plt.figure()
plt.title('Empirical Vs Estimated mu_noise')
sns.regplot(x='x', y='y', data = df,label = 'a = '+str(int(a*1000)/1000)+', b = '+str(int(b*1000)/1000)+', r = '+str(int(r*1000)/1000)) # plot data + linear regression fit
plt.legend()
plt.show()
plt.plot()

df = {'x':noise_parameters_lists['_noPrior'][:,1], 'y':noise_parameters_lists['_withPrior00'][:,1]}
regr = linear_model.LinearRegression()
regr.fit(np.array([[elmt] for elmt in df['x']]),np.array([[elmt] for elmt in df['y']]))
a,b,r = regr.coef_, regr.intercept_, regr._residues
plt.figure()
plt.title('Empirical Vs Estimated std_noise')
sns.regplot(x='x', y='y', data = df,label = 'a = '+str(int(a*1000)/1000)+', b = '+str(int(b*1000)/1000)+', r = '+str(int(r*1000)/1000)) # plot data + linear regression fit
plt.legend()
plt.show()
plt.plot()

# 4th (threshsnr)
threshsnr_df = pandas.DataFrame({'imodel':threshsnrs_lists[0],'threshsnr':threshsnrs_lists[1]})
threshsnr_df = threshsnr_df.drop(list(threshsnr_df[np.abs(threshsnr_df.threshsnr) > 100].index))
plt.figure()
plt.title('trheshsnr per model')
sns.stripplot(x = 'imodel', y = 'threshsnr',data = threshsnr_df)
plt.show()

# 5th (k_high)
k_high_df = pandas.DataFrame({'imodel':k_high_lists[0],'k_high':k_high_lists[1]})
k_high_df = k_high_df.drop(list(k_high_df[np.abs(k_high_df.k_high) > 20].index))
plt.figure()
plt.title('k_high per model')
sns.stripplot(x = 'imodel', y = 'k_high',data = k_high_df)
plt.show()
