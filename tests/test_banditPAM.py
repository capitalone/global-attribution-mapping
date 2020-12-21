#!/usr/bin/env python
import time

import numpy as np
import pandas as pd

from gam.clustering import KMedoids
from gam.spearman_distance import spearman_squared_distance

np.random.seed(42)

def test_banditPAM():
    # load the data
    df = pd.read_csv("tests/banditPAM_data.csv")
    attributions = df.values

    """"Run kmedoids on sample attributions"""
    kmed2 = KMedoids(
        4,
        dist_func="euclidean",
        # dist_func=spearman_squared_distance,
        max_iter=20,
        tol=0.001,
        init_medoids="bandit",
        swap_medoids="bandit",
        verbose=False,
    )
    start_time = time.time()
    kmed2.fit(attributions, verbose=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished test in {elapsed_time:.2f}")
    print(kmed2.centers)


    # if testing with 'euclidean' distance
    assert( kmed2.centers == [256, 209, 470, 304])

    # if testing with spearman
    # assert kmed2.centers == [526, 529, 623, 542]
