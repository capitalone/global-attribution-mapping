"""
Implementation of kmedoids using custom distance metric
Originally adapted from https://raw.githubusercontent.com/shenxudeu/K_Medoids/master/k_medoids.py
FastPAM1 from: https://arxiv.org/pdf/2008.05171.pdf
Bandit PAM from: https://arxiv.org/pdf/2006.06856.pdf
"""
import math
import sys
import time
from copy import deepcopy

import dask.array as da
import dask_distance
import matplotlib.pyplot as plt
import numpy as np
from dask_ml.metrics.pairwise import pairwise_distances as dask_pairwise_distances
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

from itertools import product


def update(existingAggregate, new_values):
    """Batch updates mu and sigma for bandit PAM using Welford's algorithm
    Refs:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    """

    (count, mean, m2) = existingAggregate
    count += len(new_values)
    # newvalues - oldMean
    delta = new_values - mean
    mean += np.sum(delta / count)
    # newvalues - newMean
    delta2 = new_values - mean
    m2 += np.sum(delta * delta2)

    return (count, mean, m2)


def finalize(existingAggregate):
    (count, mean, m2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, m2 / count, m2 / (count - 1))
    if count < 2:
        return float("nan")
    else:
        return (mean, variance, sampleVariance)


def _get_random_centers(n_clusters, n_samples):
    """Return random points as initial centers"""
    init_ids = []
    while len(init_ids) < n_clusters:
        _ = np.random.randint(0, n_samples)
        if _ not in init_ids:
            init_ids.append(_)
    return init_ids


def search_singles(X, solution_ids, dist_func, d_nearest):
    """ Inner loop for pam build and bandit build functions """
    td = float("inf")
    for j in solution_ids:
        d = cdist(X, X[j, :].reshape(1, -1), metric=dist_func).squeeze()
        tmp_delta = d - d_nearest
        g = np.where(tmp_delta > 0, 0, tmp_delta)
        tmp_td = np.sum(g)
        if tmp_td < td:
            td = tmp_td
            idx_best = j
            d_best = np.copy(d).reshape(-1, 1)

    return idx_best, d_best


def _init_pam_build(X, n_clusters, dist_func):
    """PAM BUILD routine for intialization
    Greedy allocation of medoids.  1st medoid is most central point.
    Second medoid decreases TD (total distance/dissimilarity) the most...
    ...and on until you have found all k pts
    Run time O(kn^2)
    """

    n_samples = X.shape[0]
    centers = np.zeros((n_clusters), dtype="int")
    D = np.empty((n_samples, 1))  # will append columns as we need/find them

    # find first medoid - the most central point
    print("BUILD: Initializing first medoid - ")
    td = float("inf")
    for j in range(n_samples):
        d = cdist(X, X[j, :].reshape(1, -1), metric=dist_func).squeeze()
        tmp_td = d.sum()
        if tmp_td < td:
            td = tmp_td
            centers[0] = j
            D = d.reshape(-1, 1)

    print(f"Found first medoid = {centers[0]}")

    # find remaining medoids
    print("Initializing other medoids - ")
    for i in range(1, n_clusters):
        d_nearest = np.partition(D, 0)[:, 0]
        print(i, d_nearest.min(), d_nearest.max())
        # available candidates
        unselected_ids = np.arange(n_samples)
        unselected_ids = np.delete(unselected_ids, centers[0:i])
        centers[i], d_best = search_singles(X, unselected_ids, dist_func, d_nearest)
        D = np.concatenate((D, d_best), axis=1)
        print(f"updated centers - {centers}")
    return centers


def _swap_pam(X, centers, dist_func, max_iter, tol, verbose):
    done = False
    n_samples = X.shape[0]
    n_clusters = len(centers)
    current_iteration = 1

    while not done and (current_iteration < max_iter):
        d = cdist(X, X[centers, :], metric=dist_func)
        # cache nearest (D) and second nearest (E) distances to medoids
        tmp = np.partition(d, 1)
        D = tmp[:, 0]
        E = tmp[:, 1]

        # debugging test to check that D ≤ E
        # assert np.all(E - D >= 0)

        Tih_min = float("inf")

        done = True  # let's be optimistic we won't find a swap
        for i in range(n_clusters):
            d_ji = d[:, i]
            unselected_ids = np.arange(n_samples)
            unselected_ids = np.delete(unselected_ids, centers[0:i])
            for h in unselected_ids:
                d_jh = cdist(X, X[h, :].reshape(1, -1), metric=dist_func).squeeze()

                #                def search_pairs(i, h, d, X, dist_func):/b
                # calculate K_jih
                K_jih = np.zeros_like(D)
                # if d_ji > D:
                #    Kjih = min(d(j, h) − Dj, 0)
                diff_ji = d_ji - D
                idx = np.where(diff_ji > 0)

                # K_jih[idx] = min(diff_jh[idx], 0)
                diff_jh = d_jh - D
                K_jih[idx] = np.minimum(diff_jh[idx], 0)

                # if d_ji = Dj:
                #    Kjih = min(d(j, h), Ej) − Dj
                idx = np.where(diff_ji == 0)
                K_jih[idx] = np.minimum(d_jh[idx], E[idx]) - D[idx]

                Tih = np.sum(K_jih)

                if Tih < Tih_min:
                    Tih_min = Tih
                    i_swap = i
                    h_swap = h
        # execute the swap
        if Tih_min < 0 and abs(Tih_min) > tol:
            done = False  # sorry we found a swap
            centers[i_swap] = h_swap
            if verbose:
                print("Swapped - ", i_swap, h_swap, Tih_min)
        else:
            done = True
        # our best swap would degrade the clustering (min Tih > 0)
        current_iteration = current_iteration + 1
    return centers


def _get_distance(data1, data2):
    """example distance function"""
    return np.sqrt(np.sum((data1 - data2) ** 2))


def _get_cost(X, centers_id, dist_func):
    """Return total cost and cost of each cluster"""
    dist_mat = np.zeros((len(X), len(centers_id)))
    # compute distance matrix
    if isinstance(X, da.Array):
        d = dask_pairwise_distances(
            X, np.asarray(X[centers_id, :]), metric=dist_func, n_jobs=-1
        )
        dist_mat = d.compute()
    else:
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


def _naive_swap(X, centers, dist_func, max_iter, tol, verbose):  # noqa:C901
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
            # edge case - build found the medoids, so we need to finish up the calc...
            members, costs, tot_cost, dist_mat = _get_cost(X, centers_, dist_func)
            break
        current_iteration += 1
        print("Starting Iteration: ", current_iteration)

    return centers, members, costs, tot_cost, dist_mat


class KMedoids:
    """ "
    Main API of KMedoids Clustering

    Parameters
    --------
        n_clusters: number of clusters
        batchsize: Batchsize for grabbing each medoid
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
        batchsize,
        dist_func=_get_distance,
        max_iter=1000,
        tol=0.0001,
        init_medoids=None,
        swap_medoids=None,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol
        self.batchsize = batchsize

        self.centers = None
        self.members = None
        self.init_medoids = init_medoids
        self.swap_medoids = swap_medoids

    def fit(self, X, plotit=False, verbose=True):
        """Fits kmedoids with the option for plotting

        Args:
            X (np.ndarray): The dataset being passed in.
            plotit (bool, optional): Determining whether or not to plot the output. Defaults to False.
            verbose (bool, optional): Whether or not to print out updates on the algorithm. Defaults to True.
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

        Args:
            X (np.ndarray): The dataset to be clustered
            n_clusters (int): The number of clusters that will be created on the dataset.
            dist_func (callable): Should be either spearman_squared_distance, mergeSortDistance, or distance.
            init_medoids (None, str): Defines what algorithm to run for build.
            swap_medoids (None, str): Defines what algorithm to run for swap.
            max_iter (int, optional): Maximum possible number of run throughs before cancelling. Defaults to 1000.
            tol (float, optional): Tolerance denoting minimal acceptable amount of improvement, controls early stopping. Defaults to 0.001.
            verbose (bool, optional): Whether or not to print out updates on the algorithm. Defaults to True.

        Returns:
            centers (list): Designates index of medoid relative to X.
            members (np.ndarray): Assigning membership to each sample in X.
            costs (np.ndarray): Array of costs for each cluster.
            tot_cost (int): The total cost of the distance matrix.
            dist_mat (np.ndarray): The matrix of distances from each point to all other points in the dataset.
        """
        n_samples, _ = X.shape

        # Get initial centers
        init_start = time.time()
        if init_medoids == "build":
            init_ids = _init_pam_build(X, n_clusters, dist_func)
        elif init_medoids == "bandit":
            init_ids = self._init_bandit_build(X, n_clusters, dist_func, verbose)
        else:
            init_ids = _get_random_centers(n_clusters, n_samples)
            # init_ids = [81, 593, 193, 22]
        init_end = time.time()
        init_elapsed = init_end - init_start

        if verbose:
            print("Initial centers are ", init_ids)
            print(f"Finished init  {init_elapsed} sec.")
        init_ids = list(init_ids)

        # Find which swap method we are using
        if swap_medoids == "stop":
            print("Stop method was selected.  Exiting. clustering.py near line 251")
            print(init_ids)
            sys.exit()
        #        elif self.swap_medoids:
        #            raise NotImplementedError()
        elif swap_medoids == "bandit":
            centers = self._swap_bandit(X, init_ids, dist_func, max_iter, tol, verbose)
            members, costs, tot_cost, dist_mat = _get_cost(X, centers, dist_func)
        elif swap_medoids == "pam":
            centers = _swap_pam(X, init_ids, dist_func, max_iter, tol, verbose)
            members, costs, tot_cost, dist_mat = _get_cost(X, centers, dist_func)
        else:
            centers, members, costs, tot_cost, dist_mat = _naive_swap(
                X, init_ids, dist_func, max_iter, tol, verbose
            )
        swap_end = time.time()
        if verbose:
            swap_elapsed = swap_end - init_end
            print(f"Finished swap  {swap_elapsed} sec.")

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
    ):  # noqa:C901
        """Runs kmedoids algorithm with custom dist_func.

        Args:
            X (np.ndarray): The dataset to be clustered
            n_clusters (int): The number of clusters that will be created on the dataset.
            dist_func (callable): Should be either spearman_squared_distance, mergeSortDistance, or distance.
            init_medoids (None, str): Defines what algorithm to run for build.
            swap_medoids (None, str): Defines what algorithm to run for swap.
            max_iter (int, optional): Maximum possible number of run throughs before cancelling. Defaults to 1000.
            tol (float, optional): Tolerance denoting minimal acceptable amount of improvement, controls early stopping. Defaults to 0.001.
            verbose (bool, optional): Whether or not to print out updates on the algorithm. Defaults to True.

        Returns:
            centers (list): Designates index of medoid relative to X.
            members (np.ndarray): Assigning membership to each sample in X.
            costs (np.ndarray): Array of costs for each cluster.
            tot_cost (int): The total cost of the distance matrix.
            dist_mat (np.ndarray): The matrix of distances from each point to all other points in the dataset.
        """
        n_samples, _ = X.shape

        # Get initial centers
        if init_medoids:
            init_ids = init_medoids
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

    def _update(self, count, mean, m2, new_values):
        """Batch updates mu and sigma for bandit PAM using Welford's algorithm
        Refs:
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates

        Args:
            count (int): The number of reference points.
            mean (int): The current mean
            m2 (int): The updated mean
            new_values (np.ndarray): The distance matrix

        Returns:
            count (int): The number of reference points.
            mean (int): The old mean.
            m2 (int): The new mean.
        """
        count += len(new_values)
        # newvalues - oldMean
        delta = new_values - mean
        mean += np.sum(delta / count)
        # newvalues - newMean
        delta2 = new_values - mean
        m2 += np.sum(delta * delta2)

        return count, mean, m2

    def _finalize(self, count, m2):
        """Finding variance for each new mean

        Args:
            count (int): The number of reference points.
            m2 (int): The updated mean.

        Returns:
            variance (int): The variance of the medoids
        """
        variance = m2 / count
        sample_variance = m2 / (count - 1)
        if count < 2:
            return float("nan")
        else:
            return variance, sample_variance

    def _bandit_search_singles(self, X, dist_func, d_nearest, td, tmp_arr, j, i):
        """Inner loop for pam build and bandit build functions.

        Args:
            X (np.ndarray): The dataset to be clustered.
            dist_func (callable): The distance function
            d_nearest (np.ndarray): The distances for all of the centers
            tmp_arr (np.ndarray): the array of distances from each cluster center
            j (float): The solution ids.
            i (int): The index of the cluster.

        Returns:
            tmp_arr (np.ndarray): An array of the sum of distances from the centers.
        """
        d = cdist(X, X[j, :].reshape(1, -1), metric=dist_func).squeeze()
        tmp_delta = d - d_nearest
        g = np.where(tmp_delta > 0, 0, tmp_delta)
        tmp_td = np.sum(g)
        tmp_arr[j] = tmp_td

        return tmp_arr[j]

    def _init_bandit_build(self, X, n_clusters, dist_func, verbose):
        """Orchestrating the banditPAM build

        Args:
            X (np.ndarray): The dataset.
            n_clusters (int): The number of clusters.
            dist_func (callable): The distance function
            verbose (bool): Whether or not to print out updates

        Returns:
            centers (np.ndarray): The centers of the clusters.
        """
        n_samples = X.shape[0]
        centers = np.zeros((n_clusters), dtype="int")
        self.D = np.empty((n_samples, 1))
        np.random.seed(100)
        delta = 1.0 / (1e3 * n_samples)  # p 5 'Algorithmic details'
        lambda_centers = np.vectorize(
            lambda i: self._find_medoids(
                X, n_clusters, dist_func, centers, verbose, n_samples, delta, i
            ),
            otypes="O",
        )
        centers = lambda_centers(np.arange(n_clusters))

        return centers

    def _looping_solution_ids(
        self, X, idx_ref, dist_func, d_nearest, n_used_ref, mu_x, sigma_x, j, i
    ):
        """Iterating through all of the different solution_ids

        Args:
            X (np.ndarray): The Dataset.
            idx_ref (np.ndarray): The random indices to be tested.
            dist_func (callable): The distance function.
            d_nearest (np.ndarray): The nearest points to the centers.
            n_used_ref (int): The number of used references
            mu_x (np.ndarray): The running mean.
            sigma_x (np.ndarray): The confidence interval.
            j (int): The solution ids
            i (int): The index of the center currently trying to be found.

        Returns:
            mu_x (np.ndarray): The running mean.
            sigma_x (np.ndarray): The confidence interval.
        """
        if isinstance(X, da.Array):
            d = dask_distance.cdist(X[idx_ref, :], X[j, :].reshape(1, -1), metric=dist_func).squeeze()
            d = d.compute()
        else:
            d = cdist(X[idx_ref, :], X[j, :].reshape(1, -1), metric=dist_func).squeeze()
        
        if i == 0:
            td = d.sum()
            var = sigma_x[j] ** 2 * n_used_ref
            n_used_ref, mu_x[j], var = self._update(n_used_ref, mu_x[j], var, d)
            var, var_sample = self._finalize(n_used_ref, var)
            sigma_x[j] = np.sqrt(var)
        else:
            tmp_delta = d - d_nearest[idx_ref]
            g = np.where(tmp_delta > 0, 0, tmp_delta)
            td = np.sum(g)
            mu_x[j] = ((n_used_ref * mu_x[j]) + td) / (n_used_ref + self.batchsize)
            sigma_x[j] = np.std(g)

        return sigma_x[j], mu_x[j]

    def _find_medoids(
        self, X, n_clusters, dist_func, centers, verbose, n_samples, delta, i
    ):
        """Finding all of the medoids

        Args:
            X (np.ndarray): The Dataset.
            n_clusters (int): The number of clusters.
            dist_func (callable): The distance function.
            centers (np.ndarray): The centers of the different clusters
            verbose (bool): Print out updates
            n_samples (int): The number of samples in the dataset.
            delta (float): The threshold determining whether or not a value is going to be a part of a cluster.
            i (int): The index of the center

        Returns:
            centers (np.ndarray): The list of centers for the different clusters.
        """
        mu_x = np.zeros((n_samples))
        sigma_x = np.zeros((n_samples))
        d_nearest = np.partition(self.D, 0)[:, 0]

        # available candidates - S_tar - we draw samples from this population
        unselected_ids = np.arange(n_samples)
        unselected_ids = np.delete(unselected_ids, centers[0:i])
        # solution candidates - S_solution
        solution_ids = np.copy(unselected_ids)
        n_used_ref = 0
        while (n_used_ref < n_samples) and (solution_ids.shape[0] > 1):
            # sample a batch from S_ref (for init, S_ref = X)
            idx_ref = np.random.choice(
                unselected_ids, size=self.batchsize, replace=True
            )
            ci_scale = math.sqrt(
                (2 * math.log(1.0 / delta)) / (n_used_ref + self.batchsize)
            )
            lmbda = np.vectorize(
                lambda j: self._looping_solution_ids(
                    X,
                    sorted(idx_ref),
                    dist_func,
                    d_nearest,
                    n_used_ref,
                    mu_x,
                    sigma_x,
                    j,
                    i,
                ),
                otypes="O",
            )
            lmbda(solution_ids)

            # Remove pts that are unlikely to be a solution
            C_x = ci_scale * sigma_x
            ucb = mu_x + C_x

            # check if LCB of target is <= UCB of current best
            lcb_target = mu_x - C_x
            ucb_best = ucb.min()
            solution_ids = np.where(lcb_target <= ucb_best)[0]

            # clean up any center idx that crept in...
            for ic in centers:
                if ic in solution_ids:
                    solution_ids = np.delete(solution_ids, int(ic))

            n_used_ref = n_used_ref + self.batchsize

        # finish search over the remaining candidates
        if verbose:
            print(
                f"Final eval with candidates = {solution_ids.shape[0]}"
            )  # , {solution_ids}")
        if solution_ids.shape[0] == 1:
            # save the single sample as a medoid
            centers[i] = solution_ids  # probably a type error
            if isinstance(X, da.Array):
                d = dask_distance.cdist(X, X[centers[i], :].reshape(1, -1), metric=dist_func).squeeze()
                d = d.compute()
            else:
                d = cdist(X, X[centers[i], :].reshape(1, -1), metric=dist_func).squeeze()
            d_best = np.copy(d).reshape(-1, 1)
        else:  # this is fastPam build - with far fewer pts to evaluate
            tmp_arr = np.zeros((n_samples))
            td = float("inf")
            lambda_singles = np.vectorize(
                lambda j: self._bandit_search_singles(
                    X, dist_func, d_nearest, td, tmp_arr, j, i
                ),
                otypes="O",
            )
            tmp_arr = lambda_singles(solution_ids)
            idx = np.argmin(tmp_arr)
            centers[i] = solution_ids[idx]
            if isinstance(X, da.Array):
                d_best = (
                    dask_distance.cdist(X, X[centers[i], :].reshape(1, -1), metric=dist_func)
                    .squeeze()
                    .reshape(-1, 1)
                )
                d_best = d_best.compute()
            else:
                d_best = (
                    cdist(X, X[centers[i], :].reshape(1, -1), metric=dist_func)
                    .squeeze()
                    .reshape(-1, 1)
                )
            d_best = (
                cdist(X, X[centers[i], :].reshape(1, -1), metric=dist_func)
                .squeeze()
                .reshape(-1, 1)
            )
        if i == 0:
            self.D = d_best
        else:
            self.D = np.concatenate((self.D, d_best), axis=1)
        print("\t updated centers - ", centers)

        return centers[i]

    def _swap_pairs(
        self,
        X,
        d,
        a_swap,
        dist_func,
        idx_ref,
        n_used_ref,
        mu_x,
        sigma_x,
        D,
        E,
        Tih_min,
        h_i,
    ):
        """Checking to see if there are any better center points.

        Args:
            X (np.ndarray): The Dataset.
            d (np.ndarray): distance matrix
            a_swap (tuple): Tuple of clusters as a combination of cluster index and dataset index. E.g. [[0,0],[0,1],[0,2],[1,0]...]
            dist_func (callable): distance function
            idx_ref (np.ndarray): The random indices to be tested.
            n_used_ref (int): Number of used reference points
            mu_x (np.ndarray): The Running mean.
            sigma_x (np.ndarray): The confidence interval.
            D (np.ndarray): Nearest distance to medoid
            E (np.ndarray): Second nearest distance to medoid
            Tih_min (float): The sum of values of the best medoid.
            h_i (str): Determining whether or not to find the updated mean and confidence interval or best medoid

        Returns:
            mu_x (np.ndarray): The Running mean.
            sigma_x (np.ndarray): The confidence interval.
            Tih (float): The best medoid.
        """
        h = a_swap[0]
        i = a_swap[1]
        d_ji = d[:, i]

        if h_i == "h":
            if isinstance(X, da.Array):
                d_jh = dask_distance.cdist(
                    X[idx_ref, :], X[h, :].reshape(1, -1), metric=dist_func
                ).squeeze()
                d_jh = d_jh.compute()
            else:
                d_jh = cdist(
                    X[idx_ref, :], X[h, :].reshape(1, -1), metric=dist_func
                ).squeeze()
            K_jih = np.zeros(self.batchsize)
            diff_ji = d_ji[idx_ref] - D[idx_ref]
            idx = np.where(diff_ji > 0)

            diff_jh = d_jh - D[idx_ref]
            K_jih[idx] = np.minimum(diff_jh[idx], 0)

            idx = np.where(diff_ji == 0)
            K_jih[idx] = np.minimum(d_jh[idx], E[idx]) - D[idx]

            # base-line update of mu and sigma
            mu_x[h, i] = ((n_used_ref * mu_x[h, i]) + np.sum(K_jih)) / (
                n_used_ref + self.batchsize
            )
            sigma_x[h, i] = np.std(K_jih)

            return mu_x, sigma_x

        if h_i == "i":
            if isinstance(X, da.Array):
                d_jh = dask_distance.cdist(X, X[h, :].reshape(1, -1), metric=dist_func).squeeze()
                d_jh = d_jh.compute()
            else:
                d_jh = cdist(X, X[h, :].reshape(1, -1), metric=dist_func).squeeze()

            # calculate K_jih
            K_jih = np.zeros_like(D)
            # if d_ji > D:
            #    Kjih = min(d(j, h) − Dj, 0)
            diff_ji = d_ji - D
            idx = np.where(diff_ji > 0)

            # K_jih[idx] = min(diff_jh[idx], 0)
            diff_jh = d_jh - D
            K_jih[idx] = np.minimum(diff_jh[idx], 0)

            # if d_ji = Dj:
            #    Kjih = min(d(j, h), Ej) − Dj
            idx = np.where(diff_ji == 0)
            K_jih[idx] = np.minimum(d_jh[idx], E[idx]) - D[idx]

            Tih = np.sum(K_jih)

            return Tih

    def _swap_bandit(self, X, centers, dist_func, max_iter, tol, verbose):
        """BANDIT SWAP - improve medoids after initialization
           Recast as a stochastic estimation problem
           Run time O(nlogn)
           https://arxiv.org/pdf/2006.06856.pdf

        Args:
            X (np.ndarray): The dataset.
            centers (np.ndarray): The center medoids of the different clusters
            dist_func (callable): The distance function
            max_iter (int): Max number of times to check for a better medoid.
            tol (float): Tolerance denoting minimal acceptable amount of improvement, controls early stopping.
            verbose (bool): Determining whether or not to print out updates

        Returns:
            centers (np.ndarray): The updated center medoids
        """
        done = False
        n_samples = X.shape[0]
        n_clusters = len(centers)
        current_iteration = 1
        Tih_min = float("inf")

        delta = 1.0 / (1e3 * n_samples)  # p 5 'Algorithmic details'

        while not done and (current_iteration < max_iter):
            # initialize mu and sigma
            mu_x = np.zeros((n_samples, n_clusters))
            sigma_x = np.zeros((n_samples, n_clusters))

            done = True  # let's be optimistic we won't find a swap
            
            if isinstance(X, da.Array):
                d = dask_distance.cdist(X, X[centers, :], metric=dist_func)
                d = d.compute()
            else:
                d = cdist(X, X[centers, :], metric=dist_func)
            
            # cache nearest (D) and second nearest (E) distances to medoids
            tmp = np.partition(d, 1)
            D = tmp[:, 0]
            E = tmp[:, 1]

            unselected_ids = np.arange(n_samples)
            unselected_ids = np.delete(unselected_ids, centers)

            # this needs to be the product of k x unselected_ids
            swap_pairs = np.array(
                list(product(unselected_ids, range(n_clusters))), dtype="int"
            )

            n_used_ref = 0
            while (n_used_ref < n_samples) and (swap_pairs.shape[0] > 1):
                # sample a batch from S_ref (for init, S_ref = X)
                idx_ref = np.random.choice(
                    unselected_ids, size=self.batchsize, replace=True
                )

                ci_scale = math.sqrt(
                    (2 * math.log(1.0 / delta)) / (n_used_ref + self.batchsize)
                )
                np.apply_along_axis(
                    lambda a_swap: self._swap_pairs(
                        X,
                        d,
                        a_swap,
                        dist_func,
                        sorted(idx_ref),
                        n_used_ref,
                        mu_x,
                        sigma_x,
                        D,
                        E,
                        Tih_min,
                        "h",
                    ),
                    1,
                    swap_pairs,
                )

                # downseslect mu and sigma to match candidate pairs
                flat_indices = np.ravel_multi_index(
                    (swap_pairs[:, 0], swap_pairs[:, 1]), (n_samples, n_clusters)
                )
                tmp_mu = mu_x.flatten()[flat_indices]
                tmp_sigma = sigma_x.flatten()[flat_indices]
                C_x = ci_scale * tmp_sigma

                # Remove pts that cannot be a solution - in terms of potential reward
                ucb = tmp_mu + C_x
                idx = np.argmin(ucb)
                ucb_best = ucb.min()

                # check if LCB of target is <= UCB of current best
                lcb_target = tmp_mu - C_x

                # tmp_ids = np.where(lcb_target <= ucb_best)[0]
                tmp_ids = np.where(lcb_target <= ucb_best)[0]
                swap_pairs = swap_pairs[tmp_ids]
                print("\tremaining candidates - ", tmp_ids.shape[0])  # , tmp_ids)

                n_used_ref = n_used_ref + self.batchsize
            #
            # with reduced number of candidates - run PAM swap
            # TODO - unify full swaps - like was done with search_singles
            #
            print(
                f"Entering swap with {swap_pairs.shape[0]} candidates...pts used = {n_used_ref}"
            )

            done = True  # let's be optimistic we won't find a swap
            Tih = np.apply_along_axis(
                lambda a_swap: self._swap_pairs(
                    np.array(X),
                    d,
                    a_swap,
                    dist_func,
                    sorted(idx_ref),
                    n_used_ref,
                    mu_x,
                    sigma_x,
                    D,
                    E,
                    Tih_min,
                    "i",
                ),
                1,
                swap_pairs,
            )

            idx = np.argmin(Tih)
            Tih_min = Tih[idx]
            h_swap = swap_pairs[idx][0]
            i_swap = swap_pairs[idx][1]

            if Tih_min < 0 and abs(Tih_min) > tol:
                if verbose:
                    print("\tSwapped - ", centers[i_swap], h_swap, Tih_min)
                done = False  # sorry we found a swap
                centers[i_swap] = h_swap
                print("Centers after swap - ", centers)
            else:
                done = True
                print("\tNO Swap - ", i_swap, h_swap, Tih_min)
            # our best swap would degrade the clustering (min Tih > 0)
            current_iteration = current_iteration + 1
        return centers
