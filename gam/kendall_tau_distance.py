"""
Implementation of Kendall's Tau as a pairwise distance metric
Base on
http://www.plhyu.com/administrator/components/com_jresearch/files/publications/Mixtures_of_weighted_distance-based_models_for_ranking_data_with_applications_in_political_studies.pdf

TOOD:
- consider optimizing using numpy
"""


def ktau_weighted_distance(r_1, r_2):
    """
    Computes a weighted kendall tau distance. Runs in O(n^2)
    Args:
        r_1, r_2 (list): list of weighted rankings.
                       Index corresponds to an item and the value is the weight
                       Entries should be positive and sum to 1
                       Example: r_1 = [0.1, 0.2, 0.7]
    Returns: float >= 0 representing the distance between the rankings
    """
    # confirm r_1 and r_2 have same lengths
    if len(r_1) != len(r_2):
        raise ValueError("rankings must contain the same number of elements")
    distance = 0

    for i in range(len(r_1) - 1):
        for j in range(i, len(r_1)):
            r_1_order = r_1[i] - r_1[j]
            r_2_order = r_2[i] - r_2[j]
            # check if i and j appear in a different order
            if r_1_order * r_2_order < 0:
                weight = r_1[i] * r_1[j] * r_2[i] * r_2[j]
                distance += 1 * weight
    return distance


def prep_data(X, Y):
    '''
    Data preprocessing step prior to doing merge sort/discovery of inversions
    1. Zip together x, y, and integer index
    2 Sort on X, and in the case of ties, sort on Y
    3. Remove X from the tuples (the first entry) since it's sorted we don't need to carry it
    4. return a list of (y_i, ind_i) for discovery of inversions

    Inputs:
        X, Y - two 1D array of weighted rankings
    Returns:
        yRankings - list of tuples containing
    '''
    assert(len(X) == len(Y))

    originalInd = list(range(len(X)))
    zipRankings = list(zip(X, Y, originalInd))
#    print('zipped rankings - ', zipRankings)

    # sort first by X, then by Y (secondary sort if items in X are equal)
#    zipRankings.sort(key=lambda item: (item[0], item[1]))
    zipRankings.sort(key=lambda item: (item[0]))
#    print('zipped sorted - ', zipRankings)
    # since X is sorted now (see above) - we can strip it off, and concentrate on (Y,index)
    yRankings = [(aTuple[1], aTuple[2]) for aTuple in zipRankings]
    return yRankings


def mergeSortInversions(arr, indList):
    '''
    Find inversions in 1D array of weighted rankings
    Inputs:  arr is 1D array of tuples (y_i, ind)
             indList - list of pairs of indices (used to get weights in distance calc)
    Returns:
        c - sorted array (not used, could clean up)
        inversions - interger count of inverted pairs
        indList - list of tuples containing indices (based on original list) of inverted pairs
    '''

    if len(arr) == 1:
        return arr, 0, indList
    else:
        midpt = int(len(arr) / 2)
        a = arr[:midpt]
        b = arr[midpt:]
        a, ai, indList = mergeSortInversions(a, indList)
        b, bi, indList = mergeSortInversions(b, indList)
        c = []
        i = 0
        j = 0
        inversions = 0 + ai + bi
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][0]:
            c.append(a[i])
            i += 1
            tmpInvList = []
        else:
            c.append(b[j])
            inversions += (len(a) - i)
            tmpInvList = [(a[i][1], b[j][1])]
#            print('inversion - i=:', i, ' j=', j, 'inv=', inversions, ' ind=', indList,
#                  'arr=', arr, a, b, 'a=', a[i][0], 'b=', b[j][0], 'tmpList = ', tmpInvList)

            j += 1
            indList.extend(tmpInvList)

    c += a[i:]
    c += b[j:]
    return c, inversions, indList


def distance_calc(x, y, indList):
    '''
    For use with merge sort discovered inversions
    Inputs:
       x &  y - 1D arrays of weights
       indList - list of tuples containing pairs of inversions
     Returns:
       d: calculated distance (scalar float)
     '''
#    print('input x = ', x)
#    print(' input y = ', y)
#    print('input indList - ', indList)
    d = 0
    for aTuple in indList:
        ind0 = aTuple[0]
        ind1 = aTuple[1]
        # conforms with logic in original code
        if (x[ind0] != x[ind1]) & (y[ind0] != y[ind1]):
            d += x[ind0] * x[ind1] * y[ind0] * y[ind1]
    return d


def mergeSortDistance(r1, r2):
    '''
    Utility function wrapping preprocessing, merge sort, and distance calc into 1 routine
    Inputs:
        r1, r2 - 1D arrays of rankings
    Returns:
        dist - kendall-tau distance
        inv - number of inversions found
    '''
    y_to_rank = prep_data(r1, r2)
#    print('List to sort - ', y_to_rank)
    indList = []
    c, inv, indList = mergeSortInversions(y_to_rank, indList)
    dist = distance_calc(r1, r2, indList)
#    print()
#    print('Sorted list - ', c)
#    print('Index list - ', indList)
    return dist


def pairwise_distance_matrix(rankings, dask=False):
    from sklearn.metrics import pairwise_distances
    import numpy as np
    from dask_ml.metrics.pairwise import pairwise_distances as dask_pairwise_distances
    import dask.array as da
    if dask:
        D = dask_pairwise_distances(da.array(rankings), rankings, metric=mergeSortDistance)
    else:
        D = pairwise_distances(rankings, rankings, metric=mergeSortDistance)
    return D

def pairwise_distance_matrix_legacy(rankings):
    """
    Computes a matrix of pairwise distance
    Args:
        rankings (list): each element is a list of weighted rankings (see ktau_weighted_distance)
    Returns: matrix (list of lists) containing pairwise distances
    """
    D = []
    for r_1 in rankings:
        row = []
        for r_2 in rankings:
            # original method ~ O(n^2)
            # distance = ktau_weighted_distance(r_1, r_2)

            # updated method - using merge sort ~ O(nlogn)
            distance = mergeSortDistance(r_1, r_2)
            row.append(distance)
        D.append(row)
    return D

