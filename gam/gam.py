"""
Main driver for generating global attributions

TODO:
- add integration tests
- expand to use other distance metrics
"""

import math
import csv
from collections import Counter

import numpy as np
import matplotlib.pylab as plt

from gam.clustering import KMedoids
from gam.spearman_distance import spearman_squared_distance


class GAM:
    """Generates global attributions

    Args:
        attributions_path (str): path for csv containing local attributions
        distance (str): distance metric used to compare attributions
        k (int): number of subpopulations to surface
    """

    def __init__(self, attributions_path="local_attributions.csv", distance="spearman", k=2):
        self.attributions_path = attributions_path
        self.distance = distance
        self.k = k

        self.attributions = None
        self.normalized_attributions = None
        self.feature_labels = None

        self.subpopulations = None
        self.subpopulation_sizes = None
        self.explanations = None

    def _read_local(self):
        """
        Reads attribution values and feature labels from csv

        Returns:
            attributions (numpy.ndarray): for example, [(.2, .8), (.1, .9)]
            feature labels (tuple of labels): ("height", "weight")
        """

        self.attributions = np.genfromtxt(
            self.attributions_path,
            dtype=float,
            delimiter=',',
            skip_header=1)

        with open(self.attributions_path) as attribution_file:
            self.feature_labels = next(csv.reader(attribution_file))

    @staticmethod
    def normalize(attributions):
        """
        Normalizes attributions by via absolute value
            normalized = abs(a) / sum(abs(a))

        Args:
            attributions (numpy.ndarray): for example, [(2, 8), (1, 9)]

        Returns: normalized attributions (numpy.ndarray). For example, [(.2, .8), (.1, .9)]
        """
        # keepdims for division broadcasting
        total = np.abs(attributions).sum(axis=1, keepdims=True)

        return np.abs(attributions) / total

    def _cluster(self, distance_function=spearman_squared_distance, max_iter=1000, tol=0.0001):
        """Calls kmedoids module to group attributions"""
        clusters = KMedoids(self.k, dist_func=distance_function, max_iter=max_iter, tol=tol)
        clusters.fit(self.normalized_attributions, verbose=False)

        self.subpopulations = clusters.members
        self.subpopulation_sizes = GAM.get_subpopulation_sizes(clusters.members)
        self.explanations = self._get_explanations(clusters.centers)

    @staticmethod
    def get_subpopulation_sizes(subpopulations):
        """Computes the sizes of the subpopulations using membership array

        Args:
            subpopulations (list): contains index of cluster each sample belongs to.
                Example, [0, 1, 0, 0].

        Returns:
            list: size of each subpopulation ordered by index. Example: [3, 1]
        """
        index_to_size = Counter(subpopulations)
        sizes = [index_to_size[i] for i in sorted(index_to_size)]

        return sizes

    def _get_explanations(self, centers):
        """Converts subpopulation centers into explanations using feature_labels

        Args:
            centers (list): index of subpopulation centers. Example: [21, 105, 3]

        Returns: explanations (list).
            Example: [[('height', 0.2), ('weight', 0.8)], [('height', 0.5), ('weight', 0.5)]].
        """
        explanations = []

        for center_index in centers:
            explanation_weights = self.normalized_attributions[center_index]
            explanations.append(list(zip(self.feature_labels, explanation_weights)))
        return explanations

    def plot(self, num_features=5, output_path_base=None, display=True):
        """Shows bar graph of feature importance per global explanation
        ## TODO: Move this function to a seperate module
        
        Args:
            num_features: number of top features to plot, int
            output_path_base: path to store plots
            display: option to display plot after generation, bool
        """
        if not hasattr(self, 'explanations'):
            self.generate()

        fig_x, fig_y = 5, num_features

        for idx, explanations in enumerate(self.explanations):
            _, axs = plt.subplots(1, 1, figsize=(fig_x, fig_y), sharey = True)

            explanations_sorted = sorted(explanations, key=lambda x: x[-1], reverse=False)[-num_features:]
            axs.barh(*zip(*explanations_sorted))
            axs.set_xlim([0,1])
            axs.set_title('Explanation {}'.format(idx+1), size=10)
            axs.set_xlabel('Importance', size=10)

            plt.tight_layout()
            if output_path_base:
                output_path = '{}_explanation_{}.png'.format(output_path_base, idx+1)
                # bbox_inches option prevents labels cutting off
                plt.savefig(output_path, bbox_inches='tight')

            if display:
                plt.show()

    def generate(self):
        """Clusters local attributions into subpopulations with global explanations"""
        self._read_local()
        self.normalized_attributions = GAM.normalize(self.attributions)
        self._cluster()
