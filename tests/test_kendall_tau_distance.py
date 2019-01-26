from gam.kendall_tau_distance import ktau_weighted_distance
from gam.kendall_tau_distance import pairwise_distance_matrix


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
    rankings = [r1, r2, r3]
    D = pairwise_distance_matrix(rankings)
    # check symmetry, within floating point rounding margin
    assert (D[0][1] - D[1][0]) < 1e-9
    # check diagonal is zero
    assert D[1][1] == 0
    assert D[2][2] == 0
    # distance between r2 and r3 is closer than r2 and r1
    assert D[1][2] < D[1][0]
