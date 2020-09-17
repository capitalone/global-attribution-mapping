"""
Implementation of kmedoids using custom distance metric
Adaped from https://raw.githubusercontent.com/shenxudeu/K_Medoids/master/k_medoids.py

TODO:
- refactor and test components of implementation
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances


def _get_random_centers(n_clusters, n_samples):
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


def _assign_pts_to_medoids(X, centers_id, dist_func):
    dist_mat = pairwise_distances(X, X[centers_id, :], metric=dist_func, n_jobs=-1)
    members = np.argmin(dist_mat, axis=1)
    return members, dist_mat


def _loss(x):
    D = squareform(pdist(x_in_cluster, metric=dist_func))
    loss = np.sum(D, axis=1)
    id = np.argmin(loss)
    return id, loss


def _get_cost(X, centers_id, dist_func):
    """Return total cost and cost of each cluster"""
    dist_mat = np.zeros((len(X), len(centers_id)))
    # compute distance matrix
    dist_mat = pairwise_distances(X, X[centers_id, :], metric=dist_func, n_jobs=-1)

    mask = np.argmin(dist_mat, axis=1)
    # members = np.argmin(dist_mat, axis=1)
    members = np.zeros(len(X))
    costs = np.zeros(len(centers_id))
    for i in range(len(centers_id)):
        mem_id = np.where(mask == i)
        # mem_id = np.where(members == i)
        members[mem_id] = i
        costs[i] = np.sum(dist_mat[mem_id, i])

    #    print("debug _get_cost - costs", costs.shape)
    #    print("debug _get_cost - mask ", mask)
    #    print("debug _get_cost - members ", members)
    return members, costs, np.sum(costs), dist_mat


def _naive_swap(X, centers, dist_func, max_iter, tol, verbose):
    n_samples, _ = X.shape
    members, costs, tot_cost, dist_mat = _get_cost(X, centers, dist_func)

    if verbose:
        print("Members - ", members.shape)
        print("Costs - ", costs.shape)
        print("Total cost - ", tot_cost)
    current_iteration, swapped = 0, True
    print("Max Iterations: ", max_iter)
    while True:
        swapped = False
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
                        swapped = True
                        if verbose:
                            print("Change centers to ", centers)
                        # self.centers = centers
                        # self.members = members
        if current_iteration > max_iter:
            if verbose:
                print("End Searching by reaching maximum iteration", max_iter)
            break
        if not swapped:
            if verbose:
                print("End Searching by no swaps")
            break
        current_iteration += 1
        print("Starting Iteration: ", current_iteration)
        return centers, members, costs, tot_cost, dist_mat


class KMedoids:
    """"
    Main API of KMedoids Clustering

    Parameters
    --------
        n_clusters: number of clusters
        dist_func : distance function
        max_iter: maximum number of iterations
        tol: tolerance
        init_medoids: {str, iterable, default=None} method of finding initial medoids
        swap_medoids: {str, default=None} str maps to method of performing swap 

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

    def __init__(
        self,
        n_clusters,
        dist_func=_get_distance,
        max_iter=1000,
        tol=0.0001,
        init_medoids=None,
        swap_medoids=None,
    ):
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol

        self.centers = None
        self.members = None
        self.init_medoids = init_medoids
        self.swap_medoids = swap_medoids

    def fit(self, X, plotit=False, verbose=True):
        """
        Fits kmedoids with the option for plotting
        """
        centers, members, _, _, _ = self.kmedoids_run_split(
            X,
            self.n_clusters,
            self.dist_func,
            self.init_medoids,
            self.swap_medoids,
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

        return

    def kmedoids_run_split(
        self,
        X,
        n_clusters,
        dist_func,
        init_medoids,
        swap_medoids,
        max_iter=1000,
        tol=0.001,
        verbose=True,
    ):
        """Runs kmedoids algorithm with custom dist_func.

        Returns: 
            centers -  list of int - designates index of medoid relative to X
            members -  rray (n_samples,) assigning membership to each sample in X
            costs -  
            tot_cost
            dist_mat
        """
        n_samples, _ = X.shape

        # Get initial centers
        if self.init_medoids:
            init_ids = self.init_medoids
        else:
            init_ids = _get_random_centers(n_clusters, n_samples)

        if verbose:
            print("Initial centers are ", init_ids)

        centers = init_ids

        # Find which swap method we are using
        if self.swap_medoids:
            raise NotImplementedError()
        else:
            centers, members, costs, tot_cost, dist_mat = _naive_swap(
                X, init_ids, dist_func, max_iter, tol, verbose
            )

        # members, costs, tot_cost, dist_mat = _get_cost(X, init_ids, dist_func)

        return centers, members, costs, tot_cost, dist_mat

    def kmedoids_run(
        self,
        X,
        n_clusters,
        dist_func,
        init_medoids,
        swap_medoids,
        max_iter=1000,
        tol=0.001,
        verbose=True,
    ):
        """Runs kmedoids algorithm with custom dist_func.

        Returns: 
            centers -  list of int - designates index of medoid relative to X
            members -  rray (n_samples,) assigning membership to each sample in X
            costs -  
            tot_cost
            dist_mat
        """

        n_samples, _ = X.shape

        # Get initial centers
        if self.init_medoids:
            init_ids = self.init_medoids
        else:
            init_ids = _get_random_centers(n_clusters, n_samples)
        if verbose:
            print("Initial centers are ", init_ids)
        centers = init_ids

        members, costs, tot_cost, dist_mat = _get_cost(X, init_ids, dist_func)
        if verbose:
            print("Members - ", members.shape)
            print("Costs - ", costs.shape)
            print("Total cost - ", tot_cost)
        current_iteration, swapped = 0, True
        print("Max Iterations: ", max_iter)
        while True:
            swapped = False
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
                            swapped = True
                            if verbose:
                                print("Change centers to ", centers)
                            self.centers = centers
                            self.members = members
            if current_iteration > max_iter:
                if verbose:
                    print("End Searching by reaching maximum iteration", max_iter)
                break
            if not swapped:
                if verbose:
                    print("End Searching by no swaps")
                break
            current_iteration += 1
            print("Starting Iteration: ", current_iteration)
        return centers, members, costs, tot_cost, dist_mat

    def predict(self, X):
        raise NotImplementedError()
