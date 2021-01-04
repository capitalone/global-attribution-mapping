"""
Implementation of kmedoids using custom distance metric
Adaped from https://raw.githubusercontent.com/shenxudeu/K_Medoids/master/k_medoids.py

TODO:
- refactor and test components of implementation
"""
import dask.array as da
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from dask_ml.metrics.pairwise import pairwise_distances as dask_pairwise_distances

def _get_init_centers(n_clusters, n_samples):
    """Return random points as initial centers"""
    init_ids = []
    while len(init_ids) < n_clusters:
        _ = np.random.randint(0, n_samples)
        if _ not in init_ids:
            init_ids.append(_)
    return init_ids


def _get_distance(data1, data2):
    """example distance function"""
    return np.sqrt(np.sum((data1 - data2) ** 2))


def _get_cost(X, centers_id, dist_func):
    """Return total cost and cost of each cluster"""
    dist_mat = np.zeros((len(X), len(centers_id)))
    # compute distance matrix
    if isinstance(X, np.ndarray):
        dist_mat = pairwise_distances(
            X, X[centers_id, :], metric=dist_func, n_jobs=-1
        )
    elif isinstance(X, da.Array):
        d = dask_pairwise_distances(
            X, np.asarray(X[centers_id, :]), metric=dist_func, n_jobs=-1
        )
        dist_mat = d.compute()


    mask = np.argmin(dist_mat, axis=1)
    members = np.zeros(len(X))
    costs = np.zeros(len(centers_id))
    for i in range(len(centers_id)):
        mem_id = np.where(mask == i)
        members[mem_id] = i
        costs[i] = np.sum(dist_mat[mem_id, i])
    return members, costs, np.sum(costs), dist_mat


class KMedoids:
    """"
    Main API of KMedoids Clustering

    Parameters
    --------
        n_clusters: number of clusters
        dist_func : distance function
        max_iter: maximum number of iterations
        tol: tolerance

    Attributes
    --------
        labels_    :  cluster labels for each data item
        centers_   :  cluster centers id
        costs_     :  array of costs for each cluster
        n_iter_    :  number of iterations for the best trail

    Methods
    -------
        fit(X): fit the model
            - X: 2-D numpy array, size = (n_sample, n_features)

        predict(X): predict cluster id given a test dataset.
    """

    def __init__(self, n_clusters, dist_func=_get_distance, max_iter=1000, tol=0.0001):
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol

        self.centers = None
        self.members = None

    def fit(self, X, plotit=False, verbose=True):
        """
        Fits kmedoids with the option for plotting
        """
        centers, members, _, _, _ = self.kmedoids_run(
            X,
            self.n_clusters,
            self.dist_func,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=verbose,
        )

        # set centers as instance attributes
        self.centers = centers
        self.members = members

        if plotit:
            _, ax = plt.subplots(1, 1)
            colors = ["b", "g", "r", "c", "m", "y", "k"]
            if self.n_clusters > len(colors):
                raise ValueError("we need more colors")

            for i in range(len(centers)):
                X_c = X[members == i, :]
                ax.scatter(X_c[:, 0], X_c[:, 1], c=colors[i], alpha=0.5, s=30)
                ax.scatter(
                    X[centers[i], 0],
                    X[centers[i], 1],
                    c=colors[i],
                    alpha=1.0,
                    s=250,
                    marker="*",
                )

    def kmedoids_run(
        self, X, n_clusters, dist_func, max_iter=1000, tol=0.001, verbose=True
    ):
        """Runs kmedoids algorithm with custom dist_func.

        Returns: centers, members, costs, tot_cost, dist_mat
        """
        # Get initial centers
        n_samples, _ = X.shape
        init_ids = _get_init_centers(n_clusters, n_samples)
        if verbose:
            print("Initial centers are ", init_ids)
        centers = init_ids
        members, costs, tot_cost, dist_mat = _get_cost(X, init_ids, dist_func)
        if verbose:
            print("Members - ", members.shape)
            print("Costs - ", costs.shape)
            print("Total cost - ", tot_cost)
        cc, swaped = 0, True
        print("Max Iterations: ", max_iter)
        while True:
            swaped = False
            for i in range(n_samples):
                if i not in centers:
                    for j in range(len(centers)):
                        centers_ = deepcopy(centers)
                        centers_[j] = i
                        members_, costs_, tot_cost_, dist_mat_ = _get_cost(
                            X, centers_, dist_func
                        )
                        if tot_cost_ - tot_cost < tol:
                            members, costs, tot_cost, dist_mat = (
                                members_,
                                costs_,
                                tot_cost_,
                                dist_mat_,
                            )
                            centers = centers_
                            swaped = True
                            if verbose:
                                print("Change centers to ", centers)
                            self.centers = centers
                            self.members = members
            if cc > max_iter:
                if verbose:
                    print("End Searching by reaching maximum iteration", max_iter)
                break
            if not swaped:
                if verbose:
                    print("End Searching by no swaps")
                break
            cc += 1
            print("Starting Iteration: ", cc)
        return centers, members, costs, tot_cost, dist_mat

    def predict(self, X):
        raise NotImplementedError()

