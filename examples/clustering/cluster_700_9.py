import time

import numpy as np
import pandas as pd

from gam.clustering import KMedoids
from gam.spearman_distance import spearman_squared_distance

np.random.seed(42)
# load the data
df = pd.read_csv("attr_round1.csv")
attributions = df.values

""""Run kmedoids on sample attributions"""
kmed2 = KMedoids(
    4,
    dist_func=spearman_squared_distance,
    max_iter=10,
    tol=0.01,
    init_medoids="build",  # with build time was 30.25
    swap_medoids="pam",
    verbose=False,
)
# attributions = np.array([(0.2, 0.8), (0.1, 0.9), (0.91, 0.09), (0.88, 0.12)])
start_time = time.time()
kmed2.fit(attributions, verbose=True)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Finished test in {elapsed_time:.2f}")
print(kmed2.centers)
# test that 2 attributions are in each cluster
# assert(sum(kmedoids_2.members) == 2)
