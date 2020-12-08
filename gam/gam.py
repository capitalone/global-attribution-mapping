"""
Main driver for generating global attributions

TODO:
- add integration tests
- expand to use other distance metrics
"""

import csv
import logging
import math
from collections import Counter

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score

from gam.clustering import KMedoids
from gam.kendall_tau_distance import mergeSortDistance
from gam.spearman_distance import spearman_squared_distance

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO
)


class GAM:
    """Generates global attributions

    Args:
        k (int): number of clusters and centroids to form, default=2
        attributions_path (str): path for csv containing local attributions
        attributions_df (pd.DataFrame, np.array, list): in-memory dataframe holding local attributions
        feature_labels
        cluster_method: None, or callable, default=None
            None - use GAM library routines for k-medoids clustering
            callable - user provided external function to perform clustering
        distance: {‘spearman’, ‘kendall’}  distance metric used to compare attributions, default='spearman'
        use_normalized (boolean): whether to use normalized attributions in clustering, default='True'
        scoring_method (callable) function to calculate scalar representing goodness of fit for a given k, default=None
        max_iter (int): maximum number of iteration in k-medoids, default=100
        tol (float): tolerance denoting minimal acceptable amount of improvement, controls early stopping, default=1e-3
    """

    def __init__(
        self,
        k=2,
        attributions_path="local_attributions.csv",
        attributions=None,
        feature_labels=None,
        cluster_method=None,
        distance="spearman",
        use_normalized=True,
        scoring_method=None,
        max_iter=100,
        tol=1e-3,
    ):

        self.attributions_path = attributions_path

        self.attributions = attributions
        self.feature_labels = feature_labels

        # self.normalized_attributions = None
        self.use_normalized = use_normalized
        self.clustering_attributions = None

        self.cluster_method = cluster_method

        self.distance = distance
        if self.distance == "spearman":
            self.distance_function = spearman_squared_distance
        elif self.distance == "kendall":
            self.distance_function = mergeSortDistance
        else:
            self.distance_function = (
                distance
            )  # assume this is  metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS

        self.k = k
        self.max_iter = max_iter
        self.tol = tol

        self.subpopulations = None
        self.subpopulation_sizes = None
        self.explanations = None

        self.scoring_method = scoring_method
        self.score = None

    def _read_df_or_list(self):
        """
        Converts attributions to numpy array and feature labels to a list if a pandas dataframe, numpy array, or list is passed in,

        Returns
            attributions (numpy.ndarray): for example, [(.2, .8), (.1, .9)]
            feature labels: ("height", "weight")
        """
        if isinstance(self.attributions, pd.DataFrame):
            self.attributions = np.asarray(self.attributions.values.tolist())
            self.feature_labels = self.attributions.columns.tolist()
        elif isinstance(self.attributions, (np.ndarray, list)) or isinstance(self.feature_labels, (np.ndarray, list)):
            if (isinstance(self.attributions, (np.ndarray, list)) and self.feature_labels is None) or (self.attributions is None and self.feature_labels is not None):
                raise ValueError("You must have both 'attributions' and 'feature_labels' if 'attributions' is not a dataframe.")
            if isinstance(self.attributions, list):
                self.attributions = np.asarray(self.attributions)
            elif isinstance(self.attributions, np.ndarray):
                self.attributions = self.attributions
            if isinstance(self.feature_labels, list):
                self.feature_labels = self.feature_labels
            elif isinstance(self.feature_labels, np.ndarray):
                self.feature_labels = self.feature_labels.tolist()
        else:
            self.attributions = None
            self.feature_labels = None

    def _read_local(self):
        """
        Reads attribution values and feature labels from csv

        Returns:
            attributions (numpy.ndarray): for example, [(.2, .8), (.1, .9)]
            feature labels (tuple of labels): ("height", "weight")
        """

        self.attributions = np.genfromtxt(
            self.attributions_path, dtype=float, delimiter=",", skip_header=1
        )

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

    def _cluster(self):
        # , distance_function=spearman_squared_distance, max_iter=1000, tol=0.0001):
        """Calls local kmedoids module to group attributions"""
        if self.cluster_method is None:
            clusters = KMedoids(
                self.k,
                dist_func=self.distance_function,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            clusters.fit(self.clustering_attributions, verbose=False)

            self.subpopulations = clusters.members
            self.subpopulation_sizes = GAM.get_subpopulation_sizes(clusters.members)
            self.explanations = self._get_explanations(clusters.centers)
        else:
            self.cluster_method(self)

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
            # explanation_weights = self.normalized_attributions[center_index]
            explanation_weights = self.clustering_attributions[center_index]
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
        if not hasattr(self, "explanations"):
            self.generate()

        fig_x, fig_y = 5, num_features

        for idx, explanations in enumerate(self.explanations):
            _, axs = plt.subplots(1, 1, figsize=(fig_x, fig_y), sharey=True)

            explanations_sorted = sorted(
                explanations, key=lambda x: x[-1], reverse=False
            )[-num_features:]
            axs.barh(*zip(*explanations_sorted))
            axs.set_xlim([0, 1])
            axs.set_title("Explanation {}".format(idx + 1), size=10)
            axs.set_xlabel("Importance", size=10)

            plt.tight_layout()
            if output_path_base:
                output_path = "{}_explanation_{}.png".format(output_path_base, idx + 1)
                # bbox_inches option prevents labels cutting off
                plt.savefig(output_path, bbox_inches="tight")

            if display:
                plt.show()

    def generate(self):
        """Clusters local attributions into subpopulations with global explanations"""
        if self.attributions is None:
            # we need to read in attributions from a CSV file, since we don't have any in memory
            self._read_local()
        else:
            self._read_df_or_list()
        if self.use_normalized:
            self.clustering_attributions = GAM.normalize(self.attributions)
        else:
            self.clustering_attributions = self.attributions
        self._cluster()
        if self.scoring_method:
            self.score = self.scoring_method(self)
