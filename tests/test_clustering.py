import pandas as pd
import numpy as np
from gam.clustering import KMedoids
from gam.spearman_distance import spearman_squared_distance


def test_kmedoids():
    """"Run kmedoids on sample attributions"""
    kmedoids_2 = KMedoids(
        2,
        batchsize=100,
        dist_func=spearman_squared_distance,
        max_iter=1000,
        tol=0.0001,
        init_medoids=None,
    )
    attributions = np.array([(0.2, 0.8), (0.1, 0.9), (0.91, 0.09), (0.88, 0.12)])
    kmedoids_2.fit(attributions, verbose=False)
    # test that 2 attributions are in each cluster
    assert sum(kmedoids_2.members) == 2

