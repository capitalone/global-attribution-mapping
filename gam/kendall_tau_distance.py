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


def pairwise_distance_matrix(rankings):
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
            distance = ktau_weighted_distance(r_1, r_2)
            row.append(distance)
        D.append(row)
    return D
