"""
Implementation of kmedoids using custom distance metric
Adaped from https://raw.githubusercontent.com/shenxudeu/K_Medoids/master/k_medoids.py

TODO:
- refactor and test components of implementation
"""
import math
import sys
import time
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


def _init_pam_build(X, n_clusters, dist_func):
    """ PAM BUILD routine for intialization 
        Greedy allocation of medoids.  1st medoid is most central point.
        Second medoid decreases TD (total distance/dissimilarity) the most...
        ...and on until you have found all k pts
        Run time O(kn^2) 
    """

    n_samples = X.shape[0]
    centers = np.zeros((n_clusters), dtype="int")
    D = np.zeros((n_samples, 1))  # will append columns as we need/find them

    # find medoids
    print("Initializing medoids - ")
    for i in range(n_clusters):
        d_nearest = np.partition(D, 0)[:, 0]
        td = float("inf")
        # available candidates
        unselected_ids = np.arange(n_samples)
        unselected_ids = np.delete(unselected_ids, centers[0:i])
        for j in unselected_ids:
            d = pairwise_distances(
                X, X[j, :].reshape(1, -1), metric=dist_func, n_jobs=-1
            ).squeeze()
            tmp_delta = d - d_nearest
            delta = np.where(tmp_delta > 0, 0, tmp_delta)  #
            tmp_td = np.sum(delta)
            #            print(j, d, d_nearest)
            if tmp_td < td:
                td = tmp_td
                centers[i] = j
                d_best = np.copy(d).reshape(-1, 1)
        # update D with new medoid
        # print(D.shape, d_best)
        D = np.concatenate((D, d_best), axis=1)
        print("updated centers - ", centers)

    return centers


def _init_bandit_build(X, n_clusters, dist_func):
    """ BANDIT BUILD routine for intialization 
        Recast as a stochastic estimation problem
        Run time O(nlogn) 
        https://arxiv.org/pdf/2006.06856.pdf
        S_tar = X \ M_el , S_ref = X , and g() = g
    """
    n_samples = X.shape[0]
    batchsize = 100
    delta = 1.0 / (1e3 * n_samples)  # p 5 'Algorithmic details'

    centers = np.zeros((n_clusters), dtype="int")
    D = np.zeros((n_samples, 1))  # will append columns as we need/find them

    # find medoids
    print("Initializing medoids - ")
    for i in range(n_clusters):
        # initialize mu and sigma
        mu_x = np.zeros((n_samples))
        sigma_x = float("inf") * np.ones((n_samples))
        C_x = np.zeros((n_samples))
        d_nearest = np.partition(D, 0)[:, 0]
        # available candidates - S_tar - we draw samples from this population
        unselected_ids = np.arange(n_samples)
        unselected_ids = np.delete(unselected_ids, centers[0:i])
        # solution candidates - S_solution
        solution_ids = np.copy(unselected_ids)
        n_used_ref = 0
        while (n_used_ref < n_samples) and (solution_ids.shape[0] > 1):
            # sample a batch from S_ref (for init, S_ref = X)
            idx_ref = np.random.choice(unselected_ids, size=batchsize, replace=True)
            for j in solution_ids:
                # look at distances from this point to a random subset (not whole set!)
                d = pairwise_distances(
                    X[idx_ref, :], X[j, :].reshape(1, -1), metric=dist_func, n_jobs=-1
                ).squeeze()
                tmp_delta = d - d_nearest[idx_ref]
                g = np.where(tmp_delta > 0, 0, tmp_delta)  #
                sigma_x[j] = np.std(g)
                td = np.sum(g)
                mu_x[j] = ((n_used_ref * mu_x[j]) + td) / (n_used_ref + batchsize)

            C_x = sigma_x * (
                math.sqrt((2 * math.log(1.0 / delta)) / (n_used_ref + batchsize))
            )

            # Remove pts that cannot be a solution
            idx = np.argmin(mu_x)
            mu_y = mu_x[idx]
            sigma_y = sigma_x[idx]
            C_y = sigma_y * (
                math.sqrt((2 * math.log(1.0 / delta)) / (n_used_ref + batchsize))
            )
            # check if LCB of target is > UCB of current best
            lcb_target = mu_x - C_x
            ucb_best = mu_y + C_y

            # print("debug mu, sigma - ", idx, mu_y, sigma_y)
            # print("debug shorten list of solutions - ", solution_ids)
            # print("debug shorten list of solutions - ", lcb_target.shape)
            solution_ids = np.where(lcb_target <= ucb_best)[0]

            # fix for re-introduction of centers to solutions...?
            # solution_ids = np.delete(solution_ids, np.where(solution_ids in centers))

            # print(
            #    f"\t Number of candidates - {solution_ids.shape[0]} , {mu_y:.2f}, {C_y:.2f}, {ucb_best:.2f}"
            #            )
            # print("\t candidates - ", solution_ids)
            #            if i == 1:
            #                print("i==1", solution_ids)
            #                sys.exit()

            n_used_ref = n_used_ref + batchsize
        #
        # so we have reduced the playing field to 1 or multiple candidates
        #
        if solution_ids.shape[0] == 1:
            # save the single sample as a medoid (either keep index, or find index of sample)
            centers[i] = solution_ids  # probably a type error
        else:  # this is fastPam build - but with far fewer pts to evaluate
            # we have more than one candidate - so lets check which one is best
            td = float("inf")
            for j in solution_ids:
                d = pairwise_distances(
                    X, X[j, :].reshape(1, -1), metric=dist_func, n_jobs=-1
                ).squeeze()
                tmp_delta = d - d_nearest
                g = np.where(tmp_delta > 0, 0, tmp_delta)  #
                tmp_td = np.sum(g)
                #            print(j, d, d_nearest)
                if tmp_td < td:
                    td = tmp_td
                    centers[i] = j
                    d_best = np.copy(d).reshape(-1, 1)
            #        print("during final search - updated with - ", i, j, td)
        D = np.concatenate((D, d_best), axis=1)
        print("\t updated centers - ", centers)

    return centers


# def _naive_swap(X, centers, dist_func, max_iter, tol, verbose):
def _swap_pam(X, centers, dist_func, max_iter, tol, verbose):
    done = False
    n_samples = X.shape[0]
    n_clusters = len(centers)
    current_iteration = 1

    while not done and (current_iteration < max_iter):
        d = pairwise_distances(X, X[centers, :], metric=dist_func, n_jobs=-1)
        # cache nearest (D) and second nearest (E) distances to medoids
        tmp = np.partition(d, 1)
        D = tmp[:, 0]
        E = tmp[:, 1]
        # print(tmp.shape, D.shape, E.shape)
        # print(centers)
        # debugging test to check that D ≤ E
        assert np.all(E - D >= 0)

        Tih_min = float("inf")

        done = True  # let's be optimistic we won't find a swap
        for i in range(n_clusters):
            d_ji = d[:, i]
            unselected_ids = np.arange(n_samples)
            unselected_ids = np.delete(unselected_ids, centers[0:i])
            for h in unselected_ids:
                d_jh = pairwise_distances(
                    X, X[h, :].reshape(1, -1), metric=dist_func, n_jobs=-1
                ).squeeze()
                # how to vectorize this?
                # calculate K_jih
                K_jih = np.zeros_like(D)
                # if d_ji > D:
                # or equivalently d_ji - D > 0
                #    Kjih = min(d(j, h) − Dj, 0)
                diff_ji = d_ji - D
                idx = np.where(diff_ji > 0)

                # min doesn't work in a vector sense...
                diff_jh = d_jh - D
                # K_jih[idx] = min(diff_jh[idx], 0)
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
        if Tih_min < 0:
            done = False  # sorry we found a swap
            centers[i_swap] = h_swap
            if verbose:
                print("Swapped - ", i_swap, h_swap, Tih_min)
        # else:
        # our best swap would degrade the clustering (min Tih > 0)
        # we might need to finalize some calculations to match other methods
        current_iteration = current_iteration + 1
    return centers


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
            # edge case - build found the medoids, so we need to finish up the calc...
            members, costs, tot_cost, dist_mat = _get_cost(X, centers_, dist_func)
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
        verbose=True,
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
        init_start = time.time()
        if self.init_medoids == "build":
            init_ids = _init_pam_build(X, n_clusters, dist_func)
        elif self.init_medoids == "bandit":
            init_ids = _init_bandit_build(X, n_clusters, dist_func)
        else:
            init_ids = _get_random_centers(n_clusters, n_samples)
        init_end = time.time()
        init_elapsed = init_end - init_start

        if verbose:
            print("Initial centers are ", init_ids)
            print(f"Finished in {init_elapsed} sec.")

        init_ids = list(init_ids)

        # Find which swap method we are using
        if self.swap_medoids == "stop":
            print("Stop method was selected.  Exiting. clustering.py near line 251")
            print(init_ids)
            sys.exit()
        #        elif self.swap_medoids:
        #            raise NotImplementedError()
        elif self.swap_medoids == "pam":
            centers = _swap_pam(X, init_ids, dist_func, max_iter, tol, verbose)
            members, costs, tot_cost, dist_mat = _get_cost(X, centers, dist_func)

        else:
            centers, members, costs, tot_cost, dist_mat = _naive_swap(
                X, init_ids, dist_func, max_iter, tol, verbose
            )

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
