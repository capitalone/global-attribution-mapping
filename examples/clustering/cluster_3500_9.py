#!/usr/bin/env python
"""
Notes on run times & expected medoids:
    k=2 [3196, 3483]
        pam build    : 197.5 s
        bandit build :  44.5 s

Should add in total runtime for (init + swap) for:
    random + baseline
    build + pam
    bandit + bandit
"""
import sys
import time

import numpy as np
import pandas as pd
import dask.array as da 
import dask.dataframe as dd

from gam.clustering import KMedoids
from gam.dask_clustering import DaskKMedoids
from gam.spearman_distance import spearman_squared_distance

np.random.seed(42)

# load the data
df = pd.read_csv("samples_3500.csv")
attributions = df.values
print(df.shape)

print(f"\n--------- running kmedoids ------------\n")

""""Run kmedoids on sample attributions"""
kmed2 = KMedoids(
    5,
    dist_func="euclidean",
    # dist_func=spearman_squared_distance,
    batchsize=100,
    max_iter=10,
    tol=0.01,
    init_medoids='bandit',
    swap_medoids="stop",
    verbose=False,
)

# attributions = np.array([(0.2, 0.8), (0.1, 0.9), (0.91, 0.09), (0.88, 0.12)])
start_time = time.time()
kmed2.fit(attributions, verbose=True)

# Initial centers are [1269 1134 660 1095 2417]
# [2022, 1134, 660, 1095, 2417]
# cluster sizes - [  73  682  387 1462  896] 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Finished test in {elapsed_time:.2f}")
print(kmed2.centers)
cluster_sizes = np.unique(kmed2.members, return_counts=True)[1]
print(f'cluster sizes - {cluster_sizes}')
# test that 2 attributions are in each cluster
# assert(sum(kmedoids_2.members) == 2)


print(f"\n--------- running dask kmedoids ------------\n")

""""Run kmedoids on sample attributions"""
kmed2 = DaskKMedoids(
    5,
    dist_func="euclidean",
    # dist_func=spearman_squared_distance,
    batchsize=100,
    max_iter=10,
    tol=0.01,
    init_medoids='bandit',
    swap_medoids="stop",
    verbose=False,
)
# attributions = np.array([(0.2, 0.8), (0.1, 0.9), (0.91, 0.09), (0.88, 0.12)])
attributions = dd.from_pandas(df, npartitions=4).to_dask_array(lengths=True)
start_time = time.time()
kmed2.fit(attributions, verbose=True)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Finished test in {elapsed_time:.2f}")
print(kmed2.centers)
cluster_sizes = np.unique(kmed2.members, return_counts=True)[1]
print(f'cluster sizes - {cluster_sizes}')
# test that 2 attributions are in each cluster
# assert(sum(kmedoids_2.members) == 2)