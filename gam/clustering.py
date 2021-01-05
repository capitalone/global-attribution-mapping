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

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def update(existingAggregate, new_values):
    """ Batch updates mu and sigma for bandit PAM using Welford's algorithm
    Refs:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    """

    (count, mean, m2) = existingAggregate
    count += len(new_values)
    # newvalues - oldMean
    delta = np.subtract(new_values, [mean] * len(new_values))
    mean += np.sum(delta / count)
    # newvalues - newMean
    delta2 = np.subtract(new_values, [mean] * len(new_values))
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
    """Return random points as initial centers
    """
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
        g = np.where(tmp_delta > 0, 0, tmp_delta)  #
        tmp_td = np.sum(g)
        if tmp_td < td:
            td = tmp_td
            idx_best = j
            d_best = np.copy(d).reshape(-1, 1)
    return idx_best, d_best


def _init_pam_build(X, n_clusters, dist_func):
    """ PAM BUILD routine for intialization
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
    i = 0
    td = float("inf")
    for j in range(n_samples):
        d = cdist(X, X[j, :].reshape(1, -1), metric=dist_func).squeeze()
        tmp_td = d.sum()
        if tmp_td < td:
            td = tmp_td
            centers[i] = j
            D = d.reshape(-1, 1)

    print(f"Found first medoid = {centers[0]}")

    # find remaining medoids
    print("Initializing other medoids - ")
    for i in range(1, n_clusters):
        d_nearest = np.partition(D, 0)[:, 0]
        print(i, d_nearest.min(), d_nearest.max())
        td = float("inf")
        # available candidates
        unselected_ids = np.arange(n_samples)
        unselected_ids = np.delete(unselected_ids, centers[0:i])
        centers[i], d_best = search_singles(X, unselected_ids, dist_func, d_nearest)
        D = np.concatenate((D, d_best), axis=1)
        print(f"updated centers - {centers}")
    return centers


def _init_bandit_build(X, n_clusters, dist_func, verbose):
    batchsize = 100
    centers, D = _find_first_medoid(X, n_clusters, dist_func, batchsize, verbose)
    centers = _find_remaining(X, n_clusters, dist_func, batchsize, centers, D, verbose)
    return centers


def _find_first_medoid(X, n_clusters, dist_func, batchsize, verbose):
    """ BANDIT BUILD routine for intialization
        Recast as a stochastic estimation problem
        Run time O(nlogn)
        https://arxiv.org/pdf/2006.06856.pdf
    """
    n_samples = X.shape[0]
    delta = 1.0 / (1e3 * n_samples)  # p 5 'Algorithmic details'

    centers = np.zeros((n_clusters), dtype="int")
    D = np.empty((n_samples, 1))  # will append columns as we need/find them

    # find first medoid - the most central point
    print("BANDIT: Initializing first medoid - ")
    i = 0
    td = float("inf")
    mu_x = np.zeros((n_samples))
    sigma_x = np.zeros((n_samples))
    C_x = np.zeros((n_samples))
    n_used_ref = 0
    solution_ids = np.arange(n_samples)
    while (n_used_ref < n_samples) and (solution_ids.shape[0] > 1):
        # sample a batch from S_ref (for init, S_ref = X)
        idx_ref = np.random.choice(solution_ids, size=batchsize, replace=True)
        ci_scale = math.sqrt((2 * math.log(1.0 / delta)) / (n_used_ref + batchsize))
        for j in solution_ids:
            d = cdist(X[idx_ref, :], X[j, :].reshape(1, -1), metric=dist_func).squeeze()
            td = d.sum()

            # for smaller n_samples - simple update rule is not adequate
            # mu_x[j] = ((n_used_ref * mu_x[j]) + td) / (n_used_ref + batchsize)
            # sigma_x[j] = np.std(td)

            # updates based on welford's algorithm
            # TODO - keep variance matrix to avoid conversion overflows
            var = sigma_x[j] ** 2 * n_used_ref
            existingAggregate = (n_used_ref, mu_x[j], var)
            updatedAggregate = update(existingAggregate, d)
            mu_x[j], var, var_sample = finalize(updatedAggregate)
            sigma_x[j] = np.sqrt(var)

        C_x = ci_scale * sigma_x
        ucb = mu_x + C_x
        # idx = np.argmin(ucb)
        lcb_target = mu_x - C_x

        ucb_best = ucb.min()
        solution_ids = np.where(lcb_target <= ucb_best)[0]
        if verbose:
            print("initial medoid - ", solution_ids.shape[0], ucb_best, n_used_ref)
        n_used_ref = n_used_ref + batchsize

    if solution_ids.shape[0] == 1:
        # save the single sample as a medoid (either keep index, or find index of sample)
        centers[i] = solution_ids  # probably a type error
        d = cdist(X, X[solution_ids, :].reshape(1, -1), metric=dist_func).squeeze()
        D = np.copy(d).reshape(-1, 1)
    else:  # this is fastPam build - with far fewer pts to evaluate
        # we have more than one candidate - so lets check which one is best
        td = float("inf")
        for j in solution_ids:
            d = cdist(X, X[j, :].reshape(1, -1), metric=dist_func).squeeze()
            tmp_td = np.sum(d)
            if tmp_td < td:
                td = tmp_td
                centers[i] = j
                D = d.reshape(-1, 1)
    print(f"Found first medoid = {centers[0]}")
    return centers, D


def _find_remaining(X, n_clusters, dist_func, batchsize, centers, D, verbose):
    # find the remaining medoids
    n_samples = X.shape[0]
    delta = 1.0 / (1e3 * n_samples)  # p 5 'Algorithmic details'
    print("Initializing other medoids - ")
    for i in range(1, n_clusters):
        td = float("inf")
        mu_x = np.zeros((n_samples))
        sigma_x = np.zeros((n_samples))
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
            ci_scale = math.sqrt((2 * math.log(1.0 / delta)) / (n_used_ref + batchsize))
            for j in solution_ids:
                # look at distances from this point to a random subset (not whole set!)
                d = cdist(
                    X[idx_ref, :], X[j, :].reshape(1, -1), metric=dist_func
                ).squeeze()
                tmp_delta = d - d_nearest[idx_ref]
                g = np.where(tmp_delta > 0, 0, tmp_delta)  #

                td = np.sum(g)
                mu_x[j] = ((n_used_ref * mu_x[j]) + td) / (n_used_ref + batchsize)
                sigma_x[j] = np.std(g)

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
                    solution_ids = np.delete(solution_ids, ic)

            n_used_ref = n_used_ref + batchsize

        # finish search over the remaining candidates
        if verbose:
            print(
                f"Final eval with candidates = {solution_ids.shape[0]}"
            )  # , {solution_ids}")
        if solution_ids.shape[0] == 1:
            # save the single sample as a medoid 
            centers[i] = solution_ids  # probably a type error
            d = cdist(X, X[solution_ids, :].reshape(1, -1), metric=dist_func).squeeze()
            d_best = np.copy(d).reshape(-1, 1)
        else:  # this is fastPam build - with far fewer pts to evaluate
            centers[i], d_best = search_singles(X, solution_ids, dist_func, d_nearest)
        D = np.concatenate((D, d_best), axis=1)
        print("\t updated centers - ", centers)

    return centers


def _swap_bandit(X, centers, dist_func, max_iter, tol, verbose):
    from itertools import product

    """ BANDIT SWAP - improve medoids after initialization
        Recast as a stochastic estimation problem
        Run time O(nlogn)
        https://arxiv.org/pdf/2006.06856.pdf
    """
    done = False
    n_samples = X.shape[0]
    n_clusters = len(centers)
    current_iteration = 1

    batchsize = 100
    delta = 1.0 / (1e3 * n_samples)  # p 5 'Algorithmic details'

    while not done and (current_iteration < max_iter):

        # initialize mu and sigma
        mu_x = np.zeros((n_samples, n_clusters))
        # sigma_x = float("inf") * np.ones((n_samples, n_clusters))
        sigma_x = np.zeros((n_samples, n_clusters))
        C_x = np.zeros((n_samples, n_clusters))

        Tih_min = float("inf")
        done = True  # let's be optimistic we won't find a swap
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
            idx_ref = np.random.choice(unselected_ids, size=batchsize, replace=True)

            ci_scale = math.sqrt((2 * math.log(1.0 / delta)) / (n_used_ref + batchsize))
            for a_swap in swap_pairs:
                h = a_swap[0]
                i = a_swap[1]

                d_ji = d[:, i]

                # distances from candidate medoid to ref pts
                d_jh = cdist(
                    X[idx_ref, :], X[h, :].reshape(1, -1), metric=dist_func
                ).squeeze()

                # calculate K_jih
                # K_jih = np.zeros_like(D)
                K_jih = np.zeros(batchsize)
                diff_ji = d_ji[idx_ref] - D[idx_ref]
                idx = np.where(diff_ji > 0)

                diff_jh = d_jh - D[idx_ref]
                K_jih[idx] = np.minimum(diff_jh[idx], 0)

                idx = np.where(diff_ji == 0)
                K_jih[idx] = np.minimum(d_jh[idx], E[idx]) - D[idx]

                # Tih = np.sum(K_jih)
                Tih = np.sum(K_jih)

                # baseline update of mu and sigma
                mu_x[h, i] = ((n_used_ref * mu_x[h, i]) + Tih) / (
                    n_used_ref + batchsize
                )
                sigma_x[h, i] = np.std(K_jih)

                # updates based on welford's algorithm
                # var = sigma_x[h, i]**2 * n_used_ref
                # existingAggregate = (n_used_ref, mu_x[h, i], var)
                # updatedAggregate = update(existingAggregate, K_jih)
                # mu_x[h, i], var, var_sample = finalize(updatedAggregate)
                # sigma_x[h, i] = np.sqrt(var)

            # downseslect mu and sigma to match candidate pairs
            # print("debug unravel - ", swap_pairs.shape)
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

            # this is the approach written up in paper
            # idx = np.argmin(tmp_mu)
            # mu_y = tmp_mu[idx]
            # sigma_y = tmp_sigma[idx]
            # C_y = ci_scale * sigma_y
            # ucb_best = mu_y + C_y

            # check if LCB of target is <= UCB of current best
            lcb_target = tmp_mu - C_x

            # tmp_ids = np.where(lcb_target <= ucb_best)[0]
            tmp_ids = np.where(lcb_target <= ucb_best)[0]
            swap_pairs = swap_pairs[tmp_ids]
            print("\tremaining candidates - ", tmp_ids.shape[0])  # , tmp_ids)

            n_used_ref = n_used_ref + batchsize
        #
        # with reduced number of candidates - run PAM swap
        # TODO - unify full swaps - like was done with search_singles
        #
        print(
            f"Entering swap with {swap_pairs.shape[0]} candidates...pts used = {n_used_ref}"
        )
        # print(swap_pairs.T)
        Tih_min = float("inf")

        done = True  # let's be optimistic we won't find a swap
        # for i in range(n_clusters):
        for a_swap in swap_pairs:
            h = a_swap[0]
            i = a_swap[1]
            d_ji = d[:, i]
            # for h in solution_ids:
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

            if Tih < Tih_min:
                Tih_min = Tih
                i_swap = i
                h_swap = h
        # execute the swap
        # if Tih_min < 0:
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


def _assign_pts_to_medoids(X, centers_id, dist_func):
    dist_mat = cdist(X, X[centers_id, :], metric=dist_func)
    members = np.argmin(dist_mat, axis=1)
    return members, dist_mat


def _loss(x, dist_func):
    D = squareform(pdist(x, metric=dist_func))
    loss = np.sum(D, axis=1)
    id = np.argmin(loss)
    return id, loss


def _get_cost(X, centers_id, dist_func):
    """Return total cost and cost of each cluster"""
    dist_mat = np.zeros((len(X), len(centers_id)))
    # compute distance matrix
    dist_mat = cdist(X, X[centers_id, :], metric=dist_func)

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
        verbose=False,
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
            init_ids = _init_bandit_build(X, n_clusters, dist_func, verbose)
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
        if self.swap_medoids == "stop":
            print("Stop method was selected.  Exiting. clustering.py near line 251")
            print(init_ids)
            sys.exit()
        #        elif self.swap_medoids:
        #            raise NotImplementedError()
        elif self.swap_medoids == "bandit":
            centers = _swap_bandit(X, init_ids, dist_func, max_iter, tol, verbose)
            members, costs, tot_cost, dist_mat = _get_cost(X, centers, dist_func)
        elif self.swap_medoids == "pam":
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
