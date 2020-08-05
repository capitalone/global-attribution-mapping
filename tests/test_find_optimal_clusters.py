import logging

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from gam import gam

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO
)

# seed helps consistent selection of initial medoids
np.random.seed(seed=42)


def test_find_optimal_clusters():
    """"Create sample attributions with 4 clusters"""
    attributions_file = "tests/output/test_opt.csv"

    ctr_coords = [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1)]
    X, y = make_blobs(
        n_samples=40,
        n_features=2,
        centers=ctr_coords,
        cluster_std=0.01,
        center_box=(0.0, 1.0),
        shuffle=True,
        random_state=42,
    )

    logging.info(f"blobs made - {X.shape},{y.shape}")
    df = pd.DataFrame(columns=["x1", "x2"], data=X)
    df.to_csv(attributions_file, index=False)

    """"Check cluster search via silhouette score"""
    g = gam.GAM(attributions_path=attributions_file, distance="spearman")
    g.get_optimal_clustering(max_clusters=4, verbose=True)

    logging.info(f"attributions file - {g.attributions_path}")
    logging.info(f"what we settled on - {g.k}")
    logging.info(f"g.silh_scores = {g.silh_scores}")

    # test that we pick '3' as the optimal number of clusters
    assert g.k == 3

    # test that we get withing 1e-3 of known silhouette scores
    baseline_scores = np.array([0.50248, 0.99856, 0.94462])

    cluster_ordered_list = sorted(g.silh_scores, key=lambda x: x[1])
    a_list = []
    [a_list.append(x[0]) for x in cluster_ordered_list]
    current_scores = np.array(a_list)

    assert np.allclose(current_scores, baseline_scores, atol=1e-3)
