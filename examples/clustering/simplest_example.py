import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gam.clustering import KMedoids
from gam.spearman_distance import spearman_squared_distance

""""Run kmedoids on sample attributions"""
kmed2 = KMedoids(
    2,
    dist_func=spearman_squared_distance,
    max_iter=10,
    tol=0.01,
    init_medoids='build',
    swap_medoids=None,
)
attributions = np.array(
    [
        (0.25, 0.75),
        (0.2, 0.8),
        (0.1, 0.9),
        (0.91, 0.09),
        (0.88, 0.12),
        (0.85, 0.15),
        (0.51, 0.49),
    ]
)
kmed2.fit(attributions, verbose=False)

print(f"centers - {kmed2.centers}")
# test that 2 attributions are in each cluster
# assert(sum(kmedoids_2.members) == 2)

plt.plot(attributions[:, 0], attributions[:, 1], "k.")
kx0 = attributions[kmed2.centers[0]][0]
ky0 = attributions[kmed2.centers[0]][1]
plt.plot(kx0, ky0, "ko")

kx1 = attributions[kmed2.centers[1]][0]
ky1 = attributions[kmed2.centers[1]][1]
plt.plot(kx1, ky1, "kd")

plt.show(block=False)
