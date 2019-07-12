import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from gam import gam


def test_find_optimal_clusters():
    """"Create sample attributions with 4 clusters"""
    attributions_file = 'test_opt.csv'
    X, y = make_blobs(n_samples=40, n_features=2, centers=4,
                      cluster_std=0.01, center_box=(0.0, 1.0),
                      shuffle=True, random_state=42)

    print('blobs made - ', X.shape, y.shape)
    df = pd.DataFrame(columns=['x1', 'x2'], data=X)
    df.to_csv(attributions_file)

    """"Check cluster search via silhouette score"""
    g = gam.GAM(attributions_path=attributions_file, distance="kendall_tau")
    g.get_optimal_clustering()

    print('attributions file - ', g.attributions_path)
    print('data size = ', g.normalized_attributions.shape)
    print('what we settled on - ', g.k)
    assert(g.k == 4)
