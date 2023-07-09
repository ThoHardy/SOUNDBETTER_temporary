# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:33:41 2023

@author: thoma

"""

import scipy.io
import os
import seedSOUNDBETTER
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from scipy.stats import pearsonr

realSubIDs = seedSOUNDBETTER.SubIDs
datapath = seedSOUNDBETTER.datapath


def savePCAEpochs(nb_dims=5,refSub='01'):
    
    '''
    Save Epochs with nb_dims first PCs instead of EEG channels. 
    The PCs don't correspond within subjects, however.
    To reorder by similarity of the PCs, call reorderPCAEpochs().
    '''

    explained_variances = []
    dict_components = {}
    list_reorderings = []
    list_eps = []
    
    for SubID in range(20):
        
        print('\n Sub'+realSubIDs[SubID])
        
        # gather data
        os.chdir(datapath + '/myEpochs_Active')
        data_ref = 'Epoch_'+realSubIDs[SubID]+'-epo.fif'
        myEpoch = mne.read_epochs(data_ref, preload=True)['snr == 4']
        myEpoch.filter(0.05,5,method='iir') # SCP
        
        # formate data, fit and apply PCA
        X = myEpoch.get_data()
        Xconcat = np.concatenate(np.array([x.T for x in X]))
        pca = PCA(n_components=nb_dims)
        pca_results = pca.fit(Xconcat)
        dict_components[realSubIDs[SubID]] = pca_results.components_
        explained_variances.append([pca_results.explained_variance_ratio_,realSubIDs[SubID]])
        PCAX = np.array([pca_results.transform(x.T).T for x in X])
        ep = mne.EpochsArray(PCAX,
                             mne.create_info(nb_dims, myEpoch.info["sfreq"], ch_types="eeg"),
                             tmin=-0.5)
        list_eps.append(ep)
        
    # reorder channels for coherence
    ref_components = dict_components[refSub]
    for SubID in range(20):
        print('\n','reorder Sub'+realSubIDs[SubID])
        
        if realSubIDs[SubID]==refSub :
            list_reorderings.append([str(k) for k in np.arange(0,nb_dims,1)])
            # Store as mne.Epoch
            ep = list_eps[SubID]
            os.chdir(datapath + '/myEpochs_Active_10PCs')
            data_ref = 'Epoch_'+realSubIDs[SubID]+'-epo.fif'
            ep.save(data_ref,overwrite=True)
            continue
        
        reordering = []
        
        components = dict_components[realSubIDs[SubID]]
        for d1 in range(nb_dims):
            
            correlations_this_dim = []
            for d2 in range(nb_dims):
                if str(d2) in reordering :
                    correlations_this_dim.append(-np.inf)
                else:
                    correlations_this_dim.append(pearsonr(ref_components[d2],components[d1])[0])
            
            # find the dimension with maximum correlation among the yet unmapped dimensions
            mapped_dim = np.argmax(correlations_this_dim)
            reordering.append(str(mapped_dim))
        
        list_reorderings.append(reordering)
        
        # reorder channels
        ep = list_eps[SubID].reorder_channels(reordering)
        
        # Store as mne.Epoch
        os.chdir(datapath + '/myEpochs_Active_10PCs')
        data_ref = 'Epoch_'+realSubIDs[SubID]+'-epo.fif'
        ep.save(data_ref,overwrite=True)
        
    return(explained_variances,list_reorderings)
    

    
    
    
TwoPCs, kmeans_clust, gmm_clust = False, False, True

# clusters that approximate Biyu He's first principal components
list_clusters_chans=[['C4','CP4','C6','CP6'],['C3','CP3','C5','CP5'],
                     ['Fz','FCz','F1','F2','FC1','FC2'],['Pz','POz''P1','P2','PO1','PO2'],
                     ['C3','C1','Cz','C2','C4']]#,
                     #['P4','PO4','PO6','C5','FC5','FC3'],['P3','PO5','PO3','C6','FC6','FC4']]

aucs = []
answers = []
list_heatmaps = []

TWOI = [505,515] # in indexes, not ms
    
    
'''
    # extrem SNR data, useful for initializing the centroids
    myEpoch_Silence, myEpoch_Loud = mne.read_epochs(data_ref, preload=True)['snr == 1'], mne.read_epochs(data_ref, preload=True)['snr == 6']
    myEpoch_Silence.filter(0.05,5,method='iir') # SCP
    myEpoch_Loud.filter(0.05,5,method='iir') # SCP
    X_Silence, X_Loud = myEpoch_Silence.get_data(), myEpoch_Loud.get_data()
    PCAX_Silence, PCAX_Loud = np.array([pca_results.transform(x.T) for x in X_Silence]), np.array([pca_results.transform(x.T) for x in X_Loud])
    mean_low, mean_high = np.mean(np.mean(PCAX_Silence[:,TWOI[0]:TWOI[1],:],1),0), np.mean(np.mean(PCAX_Loud[:,TWOI[0]:TWOI[1],:],1),0)
    cov_low, cov_high = np.cov(np.mean(PCAX_Silence[:,TWOI[0]:TWOI[1],:],1).T), np.cov(np.mean(PCAX_Loud[:,TWOI[0]:TWOI[1],:],1).T)
    try :
        prec_low, prec_high = np.linalg.inv(cov_low), np.linalg.inv(cov_high)
    except :
        prec_low, prec_high = np.linalg.pinv(cov_low), np.linalg.pinv(cov_high)
    
    # 2D plot
    if TwoPCs : 
        heardX, not_heardX = PCAX[list(myEpoch.metadata['audibility'] > 2)], PCAX[list(myEpoch.metadata['audibility'] < 3)]
        os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/ClustOnPCs')
        plt.figure()
        plt.scatter(heardX[:,-2],heardX[:,-1],color='blue',label='heard')
        plt.scatter(not_heardX[:,-2],not_heardX[:,-1],color='red',label='not heard')
        plt.xlabel('PC4')
        plt.ylabel('PC5')
        plt.legend()
        plt.savefig('PC4,5_Sub'+realSubIDs[SubID]+'.png')
        plt.close()
    
    real_labels = [int(b) for b in list(myEpoch.metadata['audibility'] > 2)]
    answers_this_sub = {}
    
    if kmeans_clust :
    
        # partial clustering
        for dim in range(nb_dims):
            kmeans = KMeans(n_clusters=2,init=np.array([mean_low[[dim]],mean_high[[dim]]]))
            kmeans.fit(np.mean(PCAX[:,TWOI[0]:TWOI[1],[dim]],1))
            labels = kmeans.labels_
            answers_this_sub[str(dim+1)] = roc_auc_score(labels,real_labels)
        
            for dim2 in range(dim+1,nb_dims):
                kmeans = KMeans(n_clusters=2,init=np.array([mean_low[[dim,dim2]],mean_high[[dim,dim2]]]))
                kmeans.fit(np.mean(PCAX[:,TWOI[0]:TWOI[1],[dim,dim2]],1))
                labels = kmeans.labels_
                answers_this_sub[str(dim+1)+str(dim2+1)] = roc_auc_score(labels,real_labels)
                
        # clustering on all dimensions
        kmeans = KMeans(n_clusters=2,init=np.array([mean_low,mean_high]))
        kmeans.fit(np.mean(PCAX[:,TWOI[0]:TWOI[1],:],1))
        labels = kmeans.labels_
        answers_this_sub['1-10'] = roc_auc_score(labels,real_labels)
        
        # plot AUCs
        os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/ClustOnPCs/KMeans')
        names = list(answers_this_sub.keys())
        values = list(answers_this_sub.values())
        plt.figure()
        plt.bar(range(len(answers_this_sub)), values, tick_label=names)
        plt.title('AUC for different KMeans-clusterings, Sub'+realSubIDs[SubID])
        plt.ylabel('AUC')
        plt.savefig('AUCs_Sub'+realSubIDs[SubID]+'_KMeans.png')
        plt.close()
        
    
    
    if gmm_clust :
    
        # 1D clustering
        answers_this_sub = {}
        for dim in range(nb_dims):
            gmm = GaussianMixture(n_components=2,covariance_type='full', means_init=np.array([mean_low[[dim]],mean_high[[dim]]]))
            gmm.fit(np.mean(PCAX[:,TWOI[0]:TWOI[1],[dim]],1))
            labels = gmm.predict(np.mean(PCAX[:,TWOI[0]:TWOI[1],[dim]],1))
            try :
                answers_this_sub[str(dim+1)] = roc_auc_score(labels,real_labels)
            except ValueError :
                answers_this_sub[str(dim+1)] = 0
                
        # plot AUCs
        os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/ClustOnPCs/GMM')
        names = list(answers_this_sub.keys())
        values = list(answers_this_sub.values())
        plt.figure()
        plt.bar(range(len(answers_this_sub)), values, tick_label=names)
        plt.title('AUC for different GMM-clusterings, Sub'+realSubIDs[SubID])
        plt.ylabel('AUC')
        plt.savefig('AUCs_Sub'+realSubIDs[SubID]+'_1DGMM.png')
        plt.close()
        
        
        # 2D clustering
        heatmap = np.diag(list(answers_this_sub.values()))
        for dim1 in range(nb_dims):
            for dim2 in range(dim1+1,nb_dims):
                gmm = GaussianMixture(n_components=2,covariance_type='full', means_init=np.array([mean_low[[dim1,dim2]],mean_high[[dim1,dim2]]]))
                gmm.fit(np.mean(PCAX[:,TWOI[0]:TWOI[1],[dim1,dim2]],1))
                labels = gmm.predict(np.mean(PCAX[:,TWOI[0]:TWOI[1],[dim1,dim2]],1))
                try :
                    heatmap[dim1,dim2], heatmap[dim2,dim1] = roc_auc_score(labels,real_labels), roc_auc_score(labels,real_labels)
                except ValueError : # if prediction unimodal
                    heatmap[dim1,dim2], heatmap[dim2,dim1] = 0, 0
        list_heatmaps.append(heatmap)
        # plot AUC heatmap
        os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/ClustOnPCs/GMM')
        plt.figure()
        ax = sns.heatmap(heatmap, linewidth=0.5,xticklabels=np.arange(1,11,1),yticklabels=np.arange(1,11,1))
        plt.title('Paiwise clusteringof the PCs, Sub'+realSubIDs[SubID])
        plt.savefig('AUCs_Sub'+realSubIDs[SubID]+'_2DGMM.png')
        plt.close()
        
        # Progressive clustering : from [PC1] to [PC1, ..., PC10]
        answers_this_sub = {}
        for dim in range(nb_dims):
            gmm = GaussianMixture(n_components=2,covariance_type='full', means_init=np.array([mean_low[np.arange(0,dim+1,1)],mean_high[np.arange(0,dim+1,1)]]))
            gmm.fit(np.mean(PCAX[:,TWOI[0]:TWOI[1],np.arange(0,dim+1,1)],1))
            labels = gmm.predict(np.mean(PCAX[:,TWOI[0]:TWOI[1],np.arange(0,dim+1,1)],1))
            try :
                answers_this_sub[str(dim+1)] = roc_auc_score(labels,real_labels)
            except ValueError :
                answers_this_sub[str(dim+1)] = 0
                
        # plot AUCs
        os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/ClustOnPCs/GMM')
        names = list(answers_this_sub.keys())
        values = list(answers_this_sub.values())
        plt.figure()
        plt.bar(range(len(answers_this_sub)), values, tick_label=names)
        plt.title('AUC for different GMM-clusterings, Sub'+realSubIDs[SubID]+', PC1to1-10')
        plt.ylabel('AUC')
        plt.savefig('AUCs_Sub'+realSubIDs[SubID]+'_PC1to1-10GMM.png')
        plt.close()
        
        
        # Progressive clustering : from [PC1, ..., PC10] to [PC10]
        answers_this_sub = {}
        for dim in range(nb_dims):
            gmm = GaussianMixture(n_components=2,covariance_type='full', means_init=np.array([mean_low[np.arange(dim,nb_dims,1)],mean_high[np.arange(dim,nb_dims,1)]]))
            gmm.fit(np.mean(PCAX[:,TWOI[0]:TWOI[1],np.arange(dim,nb_dims,1)],1))
            labels = gmm.predict(np.mean(PCAX[:,TWOI[0]:TWOI[1],np.arange(dim,nb_dims,1)],1))
            try :
                answers_this_sub[str(dim+1)] = roc_auc_score(labels,real_labels)
            except ValueError :
                answers_this_sub[str(dim+1)] = 0
                
        # plot AUCs
        os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/ClustOnPCs/GMM')
        names = list(answers_this_sub.keys())
        values = list(answers_this_sub.values())
        plt.figure()
        plt.bar(range(len(answers_this_sub)), values, tick_label=names)
        plt.title('AUC for different GMM-clusterings, Sub'+realSubIDs[SubID]+', PC1-10to10')
        plt.ylabel('AUC')
        plt.savefig('AUCs_Sub'+realSubIDs[SubID]+'_PC1-10to10GMM.png')
        plt.close()
        
        
        
        
avg_heatmap = np.mean(list_heatmaps,0)
os.chdir('C:/Users/thoma/OneDrive/Documents/Consciousness_is_Bifurcation/Exploratory_analysis/ClustOnPCs/GMM')
plt.figure()
ax = sns.heatmap(avg_heatmap, linewidth=0.5,xticklabels=np.arange(1,11,1),yticklabels=np.arange(1,11,1))
plt.title('Paiwise clusteringof the PCs, Sub'+realSubIDs[SubID])
plt.savefig('Average_AUCs_2DGMM.png')
plt.close()
    '''

    
