import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gam.spearman_distance import spearman_squared_distance
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs, make_classification
# from sklearn.metrics import silhouette_samples, silhouette_score


# load the data
k = 2
csv_file = 'attr_round1.csv'
#k = 5
#csv_file = 'samples_3500.csv'
df = pd.read_csv(csv_file)
X = df.values
print(df.shape)

plt.close("all")
np.random.seed(seed=42)


def get_init_centers(n_clusters, n_samples):
    """Return random points as initial centers"""
    init_ids = []
    while len(init_ids) < n_clusters:
        _ = np.random.randint(0, n_samples)
        if _ not in init_ids:
            init_ids.append(_)
    return init_ids


initial_medoids = get_init_centers(k, X.shape[0])
#initial_medoids = [81, 593, 193, 152]
my_func = spearman_squared_distance
my_metric = distance_metric(type_metric.USER_DEFINED, func=my_func)
#km_clusterer = kmedoids(X, initial_medoids, metric=my_metric, tolerance = 1e-5, ccore =True, iter_max=1e5, data_type='points')
km_clusterer = kmedoids(X, initial_medoids,  metric='euclidean', tolerance = 1e-9, ccore =True, iter_max=1e5, data_type='points')

start_time = time.time()
km_clusterer.process()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Finished fit in {elapsed_time} sec.")

clusters = km_clusterer.get_clusters()

centers = X[km_clusterer.get_medoids()]
print(km_clusterer.get_medoids())
