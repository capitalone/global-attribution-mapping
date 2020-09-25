import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from gam.clustering import KMedoids
from gam.spearman_distance import spearman_squared_distance

# np.random.seed(42)
# load the data
df = pd.read_csv("attr_round1.csv")
attributions = df.values

""""Run kmedoids on sample attributions"""
kmed2 = KMedoids(
    4,
    dist_func=spearman_squared_distance,
    max_iter=10,
    tol=0.01,
    init_medoids=None,  # with build time was 30.25
    swap_medoids='bandit',
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


#pca = PCA(n_components=2)
#pca.fit(attributions)

# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

#att_pca = pca.transform(attributions)
#ctr_pca = pca.transform(attributions[kmed2.centers])
#tru_ctr_pca = pca.transform(attributions[[81, 593, 193, 152]])
#plt.plot(att_pca[:, 0], att_pca[:, 1], "k.")
#plt.plot(ctr_pca[:, 0], ctr_pca[:, 1], "bo")
#plt.plot(ctr_pca[:, 0], tru_ctr_pca[:, 1], "go")
#plt.show()


# track centers and memberships...
#[81, 593, 193, 152]
#np.unique(kmed2.members, return_counts=True)
#(array([0., 1., 2., 3.]), array([355,  79, 133, 133]))
