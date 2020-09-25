import time

import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids

from gam.spearman_distance import spearman_squared_distance

# load the data
df = pd.read_csv("samples_3500.csv")
attributions = df.values
print(df.shape)

kmedoids = KMedoids(n_clusters=5, random_state=0, metric=spearman_squared_distance, max_iter=1000)
start_time = time.time()
kmedoids.fit(attributions)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Finished fit in {elapsed_time} sec.")

cluster_sizes = np.unique(kmedoids.labels_, return_counts=True)[1]

print(kmedoids.medoid_indices_)
print(cluster_sizes)


print(kmedoids.inertia_)
