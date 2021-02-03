from gam.kendall_tau_distance import pairwise_distance_matrix
from sklearn.metrics import pairwise_distances
from dask.distributed import Client
from gam.kendall_tau_distance import (ktau_weighted_distance,
                                      mergeSortDistance,
                                      pairwise_distance_matrix)
import numpy as np
import dask.array as da

def test_ktau_symmetry():
    """Checks symmetry holds for distance metric"""
    r1 = [0.1, 0.2, 0.7]
    assert ktau_weighted_distance(r1, r1) == 0


def test_ktau_relative_distances():
    """Compares relative distances of weighted rankings"""
    r1 = [0.05, 0.2, 0.7, 0.05]
    r2 = [0.23, 0.24, 0.26, 0.27]
    r3 = [0.22, 0.24, 0.26, 0.28]
    # r2 and r3 should be closer than r2 to r1
    assert ktau_weighted_distance(r2, r3) < ktau_weighted_distance(r2, r1)


def test_pairwise_distance_matrix():
    r1 = [0.05, 0.2, 0.7, 0.05]
    r2 = [0.23, 0.24, 0.26, 0.27]
    r3 = [0.22, 0.24, 0.26, 0.28]
    rankings = np.array([r1, r2, r3])
    D = pairwise_distance_matrix(rankings)

    # check symmetry, within floating point rounding margin
    assert (D[0][1] - D[1][0]) < 1e-9
    # check diagonal is zero
    assert D[1][1] == 0
    assert D[2][2] == 0
    # distance between r2 and r3 is closer than r2 and r1
    assert D[1][2] < D[1][0]


def test_dask_pairwise_distance_matrix():
    client = Client()
    r1 = [0.05, 0.2, 0.7, 0.05]
    r2 = [0.23, 0.24, 0.26, 0.27]
    r3 = [0.22, 0.24, 0.26, 0.28]
    rankings = np.array([r1, r2, r3])

    D = pairwise_distance_matrix(da.from_array(rankings))
    # check symmetry, within floating point rounding margin
    assert (D[0][1] - D[1][0]) < 1e-9
    # check diagonal is zero
    assert D[1][1] == 0
    assert D[2][2] == 0
    # distance between r2 and r3 is closer than r2 and r1
    assert D[1][2] < D[1][0]
    client.close()



def test_ktau_accuracy():
    """ Floating point accuracy test for testing faster calculation methods """
    r1 = [0.27, 0.24, 0.26, 0.23]
    r2 = [0.05, 0.2, 0.7, 0.05]
    assert ktau_weighted_distance(r1, r2) == 0.0031050000000000006
    assert mergeSortDistance(r1, r2) == 0.0031050000000000006
