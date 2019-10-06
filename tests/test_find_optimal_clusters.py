import numpy as np
import pandas as pd
import logging
from sklearn.datasets import make_blobs
from gam import gam

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO
)


def test_find_optimal_clusters():
    """"Create sample attributions with 4 clusters"""
    attributions_file = "test_opt.csv"
    X, y = make_blobs(
        n_samples=40,
        n_features=2,
        centers=4,
        cluster_std=0.01,
        center_box=(0.0, 1.0),
        shuffle=True,
        random_state=42,
    )

    logging.info(f"blobs made - {X.shape},{y.shape}")
    df = pd.DataFrame(columns=["x1", "x2"], data=X)
    df.to_csv(attributions_file)

    """"Check cluster search via silhouette score"""
    g = gam.GAM(attributions_path=attributions_file, distance="kendall_tau")
    g.get_optimal_clustering(max_clusters=6, verbose=True)

    logging.info(f"attributions file - {g.attributions_path}")
    logging.info(f"what we settled on - {g.k}")

    assert g.silh_scores == [
        (-0.6997008553383604, 4),
        (-0.49646173501281243, 3),
        (-0.20264778322315316, 6),
        (-0.15419026951164008, 5),
        (0.3760497014154821, 2),
    ]
    assert g.k == 2
