import time

import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids

from gam.spearman_distance import spearman_squared_distance

# load the data
k = 4
csv_file = 'attr_round1.csv'
#k = 5
#csv_file = 'samples_3500.csv'
df = pd.read_csv(csv_file)
attributions = df.values
print(df.shape)

# init methods are: 'heuristic', 'random', 'k-medoids++'
max_its = int(1e8)
kmedoids = KMedoids(n_clusters=k, random_state=0, init='k-medoids++', metric='euclidean', max_iter=max_its)
#kmedoids = KMedoids(n_clusters=k, random_state=0, init='k-medoids++', metric=spearman_squared_distance, max_iter=1000000)
start_time = time.time()
kmedoids.fit(attributions)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Finished fit in {elapsed_time} sec.")

cluster_sizes = np.unique(kmedoids.labels_, return_counts=True)[1]

print('indices = ', kmedoids.medoid_indices_)
#print(cluster_sizes)


#print(kmedoids.inertia_)
