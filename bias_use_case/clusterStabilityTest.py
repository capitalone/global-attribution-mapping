#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:20:30 2019

@author: jnm703

Q: how does the number of samples used during clustering influence the results?

...we can reuse this script on each use case (baseline, features, bias) to test its stability

To answer:
    1. loop over number of samples
    2. loop over number of clusters (this function just got added to GAM)
    3. Plot mediod locations (PCA then plot PCA1 vs PCA2?)
"""
import sys
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pathToGAM = '/Users/jnm703/src/global-attribution-mapping/'
sys.path.insert(0, pathToGAM)
from gam import gam



# load up all the FICO attributions
local_attributions = 'bias_deeplift_attributions-50-50-basline.csv'
df = pd.read_csv(local_attributions)
df = df.drop(labels='Unnamed: 0', axis=1)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df.values)

#sys.exit()

optimalClusterList = []
sampleCountList = [100, 200] #100, 200]

gamDict = {}
for sampleCount in sampleCountList:
    samples = df.sample(n=sampleCount, random_state=42, axis=None)
    print(samples.shape)

    # save the test samples out to CSV for gam
    attributions_file = 'samples_' + str(sampleCount).zfill(4) + '.csv'
    samples.to_csv(attributions_file)

    # borrowed from gam tests for optimal clustering
    g = gam.GAM(attributions_path=attributions_file, distance="kendall_tau")
    g.get_optimal_clustering()
    optimalClusterList.append(g.k)
    gamDict[sampleCount] = g

    print('attributions file - ', g.attributions_path)
    print('data size = ', g.normalized_attributions.shape)
    print('what we settled on - ', g.k)


    # take list of medoids, pull out values, then project thru PCA and plot PCA1 vs PCA2
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    pcaMediods =  np.zeros((g.k,2))
    for i in range(g.k):
        anExp = g.explanations[i]
        expValues = []
        for aTuple in anExp:
            if len(aTuple[0])>0:  # avoid that empty string...<need to debug that...where's it come from?>
                expValues.append(aTuple[1])

#        print( expValues )

        plotPt = pca.transform(np.asarray(expValues).reshape(1,-1))[0]
        pcaMediods[i,:] = plotPt

    plt.plot( pcaMediods[:,0], pcaMediods[:,1], 'ko')
    ax.set_xlabel('PCA1')
    ax.set_xlabel('PCA2')
    titleStr = 'Samples = ' + str(sampleCount)
    ax.set_title(titleStr)
    figFile = 'optimal_clusters_samples_' + str(sampleCount).zfill(4) + '.png'
    plt.savefig(figFile)

pickleFile = 'stability_gamDict.pkl'
with open(pickleFile, 'wb') as f:
    pickle.dump(gamDict, f)

