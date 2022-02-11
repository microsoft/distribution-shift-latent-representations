# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import os
import random
from collections import Counter
from itertools import combinations
from multiprocessing import *
from time import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sklearn.metrics
import statsmodels.stats.weightstats as ws 
from persim import PersistenceImager
from scipy.spatial import distance
from scipy.spatial.distance import cdist, pdist
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dslr.distance_utils import persistence_distance
from dslr.persistence_utils import _compute_persistence_diagram
from dslr.sampling_utils import subsample


def perturbation_tolerance_test(
    X_embeddings: np.ndarray,
    Y_embeddings: np.ndarray,
    distance_measure: Callable,
    criteria: str = 'knn_recall',
    criteria_threshold: float = None,
    sample_size: int = 1000,
    num_epochs: int = 1,
) -> Tuple[dict, float, float, float, str, bool]:
    """Tests if sample sets are within specified tolerance.

    This test finds the radius of shapes (for certain measure) around dataset X,
    subject to perturbation, that satisfies some performance criteria.

    Consider two datasets X and Y, of same embedding dimension, and not
    in 1:1 correspondence. Can we tell if they differ, based on a human-
    interpretable performance criteria?

    Approach:
    1. For growing noise, n times, compute criteria performance on (X, X_noise).
    2. Stop when median(performance) crosses threshold value, and report noise
        level, as noise_star.
    3. For each level of noise, collect the samples of performance values.
    4. Identify noise_star level, where median performance hits threshold.
    5. Compute n times, d(X, X_noise_star), and report median as dist_star.
    6. Eval whether d(X, Y) < dist_star.

    Output sensitivity has the following form:
        sensitivity = {
            noise_level_0: [criteria values at level 0],
            noise_level_1: [criteria values at level 1],
            ...
        }

    Args:
        X_embeddings: First dataset.
        Y_embeddings: Second dataset.
        distance_measure: Distance measure, callable with two arguments X and Y
            as first two args.
        criteria: Type of performance criteria.
        criteria_threshold: Max/min value, depending on criteria.
        sample_size: Max size of subsample. (Will be smaller if dataset is smaller.)
        num_epochs: Effective number of epochs to run.

    Returns:
        sensitivity: Dict, with noise level as Key, and list of performance
            values for that noise level as Value.
        distance_xy: Measure value between X and Y.
        distance_star: Distance value between X and X + noise, at threshold.
        criteria_star: Criteria value between X and X + noise, at threshold.
        decision_summary: Printable summary of decision.
        decision_binary: True if shift detected, else False.
    """

    # Validate, center, and normalize embeddings.
    assert X_embeddings.shape[1] == Y_embeddings.shape[1], \
        'Embeddings must be of same dimension'
    scaler = StandardScaler()
    scaler.fit(np.vstack((X_embeddings, Y_embeddings)))
    X_emb = scaler.transform(X_embeddings)
    Y_emb = scaler.transform(Y_embeddings)

    # Choose criteria function. Function must take in dataset X, and noise level.
    if criteria == 'knn_recall':
        criteria_fn = compute_knn_recall_with_arrays
    else:
        raise NotImplementedError

    # Compute subsamples of X and Y.
    def total_num_samples(arr):
        """Sets total number of samples to draw, based on size of dataset.
        If sample_size is >= arr.shape[0], only sample full dataset once.
        """
        if sample_size >= arr.shape[0]:
            return 1
        else:
            _sets_per_epoch = int(np.ceil(arr.shape[0] / sample_size))
            return num_epochs * _sets_per_epoch

    total_num_samples_x = total_num_samples(X_embeddings)
    total_num_samples_y = total_num_samples(Y_embeddings)
    total_num_samples = max(total_num_samples_x, total_num_samples_y)
    print(f'Num subsample sets for X, Y: {total_num_samples}')

    # Create all subsamples of X and Y, at once.
    subsamples_x = [
        subsample(X_emb, sample_n=sample_size)
        for _ in range(total_num_samples)
    ]
    subsamples_y = [
        subsample(Y_emb, sample_n=sample_size)
        for _ in range(total_num_samples)
    ]


    # Set up sampling and noise configs.
    # TODO: Confirm range/step for noise levels.
    num_criteria_samples = 3
    noise_grid_size = 10
    noise_levels = np.logspace(-2, 0, num=noise_grid_size, base=10)
    #print(f'Noise levels: {noise_levels}\n')

    def apply_noise(arr, noise_level):
        noise = np.random.normal(loc=0, scale=noise_level, size=arr.shape)
        return (arr + noise).astype(np.float32)

    # Collect sensitivity scores.
    sensitivity = {}

    for noise_level in noise_levels:
        sensitivity[noise_level] = {'criteria_values': [], 'distance_values': []}

        for subsample_x in subsamples_x:
            criteria_values = []
            distance_values = []
            for _ in range(num_criteria_samples):
                subsample_x_noisy = apply_noise(subsample_x, noise_level)
                criteria_values.append(criteria_fn(subsample_x, subsample_x_noisy))
                distance_values.append(distance_measure(subsample_x, subsample_x_noisy))

            # Store results for this noise level.
            sensitivity[noise_level]['criteria_values'].extend(criteria_values)
            sensitivity[noise_level]['distance_values'].extend(distance_values)

        # Break out of loop, if threshold is reached.
        if np.median(sensitivity[noise_level]['criteria_values']) < criteria_threshold:
            break

    # Collect criteria performance and distance, for given threshold.

    # Initialize with "failure" values, in case no pre-threshold results exist.
    criteria_star = -1.
    distance_star = -1.

    if len(sensitivity) > 1:
        # Pick noise level just before threshold was reached, i.e. second-to-last.
        # object in sensitivty.
        noise_star = list(sensitivity.keys())[-2]
        criteria_star = np.median(sensitivity[noise_star]['criteria_values'])

        # Sample distances associated with noise_star.
        distance_star = np.median(sensitivity[noise_star]['distance_values'])

    # Compute distance between X and Y.
    distances_xy = []
    for subsample_x in subsamples_x:
        for subsample_y in subsamples_y:
            distances_xy.append(distance_measure(subsample_x, subsample_y))
    distance_xy = np.median(distances_xy)

    # Summarize decision.
    if distance_star == -1:
        decision_summary = (
            'DECISION: NOT SIMILAR. None of the noise levels tested satisfied '
            'criteria threshold.\n'
        )
    else:
        if distance_xy < distance_star:
            decision = 'DECISION: No shift detected.'
            decision_binary = False
        else:
            decision = 'DECISION: Shift detected.'
            decision_binary = True
        decision_summary = (
            f'{decision}\n'
            f'criteria = "{criteria}"\n'
            f'criteria_threshold =          {criteria_threshold}\n'
            f'criteria threshold (actual) = {criteria_star:.4f}\n'
            f'distance_xy =                 {distance_xy:.4f}\n'
            f'distance threshold (actual) = {distance_star:.4f}\n'
            'Test decides "SIMILAR" if distance_xy < distance threshold (actual).\n'
        )

    return sensitivity, distance_xy, distance_star, criteria_star, decision_summary, decision_binary

def plot_perturbation_sensitivity(
    sensitivity: dict,
    distance_star: float = None,
    criteria_star: float = None,
    tag: str = None
) -> plt.Figure:
    """Plots results of perturbation_tolerance_test.

    Args:
        sensitivity: Map of noise level to associated criteria values and distance values.
        distance_star: Distance associated with actual criteria threshold.
        criteria_star: Actual criteria threshold.
        tag: Optional string to add to plot title.

    Returns:
        fig: Matplotlib figure with results.
    """
    fig, ax = plt.subplots()

    x = []
    x_tick_labels = ['0']
    y_criteria = []
    y_distance = []

    # Expand to full set of x- and y-coordinates, with x-jitter for values
    # of the same noise level.
    for i, noise_level in enumerate(sensitivity):
        x_tick_labels.append(f'{noise_level:.2f}')
        criteria_values = sensitivity[noise_level]['criteria_values']
        distance_values = sensitivity[noise_level]['distance_values']
        for j in range(len(criteria_values)):
            x.append(i + 1 + np.random.normal(scale=0.1))
            y_criteria.append(criteria_values[j])
            y_distance.append(distance_values[j])

    # Plot both sets of values.
    ax.scatter(x, y_criteria, s=15, c='blue', alpha=0.5, label='criteria')
    if criteria_star:
        ax.axhline(y=criteria_star, c='blue', alpha=0.5, label='criteria_thresh')

    # Overwrite x tick labels.
    # TODO: Troubleshoot x tick locs, to match x_tick_labels
    x_tick_locs = ax.get_xticks().tolist()
    x_tick_locs = np.linspace(min(x_tick_locs), max(x_tick_locs), len(x_tick_labels) + 1)
    ax.xaxis.set_major_locator(mticker.FixedLocator(x_tick_locs))
    ax.set_xticklabels(x_tick_labels + ['[end]'])
    # ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Criteria Value')
    _ax = ax.twinx()
    _ax.scatter(x, y_distance, s=15, c='green', alpha=0.5, label='distance')
    if distance_star:
        _ax.axhline(y=distance_star, c='green', alpha=0.5, label='distance_thresh')
    _ax.set_ylabel('Distance Value')

    # Align grid lines.
    # _ax.set_yticks(
    #     np.linspace(_ax.get_yticks()[0], _ax.get_yticks()[-1], len(ax.get_yticks()) - 2)
    # )
    _ax.grid(None)

    fig.legend(bbox_to_anchor=(1.2, 1.05))

    title = 'Perturbation Tolerance'
    if tag:
        title += f', {tag}'
    ax.set_title(title)

    return fig

def subsample_based_distribution_shift_test(
    X_embeddings: np.ndarray,
    Y_embeddings: np.ndarray,
    distance_measure: Callable,
    eps: float = 0.0,
    num_epochs: int = 1,
    sample_size: int = 1000,
    decision: str = "threshold",
    pvalue_threshold: float = 0.05
) -> Tuple[float, float, float, str, bool]:
    """
    Compares intra-dataset distribution distance to inter-dataset distribution
    distance.

    This test compares the distances between subsamples from the same dataset
    to distances between subsamples across datasets, to determine if a drift has
    occurred between the two input embedding spaces.

    Args:
        X_embeddings: First dataset.
        Y_embeddings: Second dataset.
        distance_measure: Distance measure, callable with two arguments X and Y
            as first two args.
        eps: Allowed threshold for detecting a distribution shift.
        num_epochs: Effective number of epochs to run.
        sample_size: Max size of subsample. (Will be smaller if dataset is smaller.)
        decision: outcome of the test determined by threshold or stats test ("threshold" or "stats")
        pvalue_threshold: threshold for pvalue, default 0.05

    Returns:
        D_xx: Measure value between subsamples of X.
        D_xy: Measure value between subsamples of X and Y.
        eps: Margin of acceptable difference between distances.
        decision_summary: Printable summary of decision.
        decision_binary: True if shift detected, else False.
    """

    # Validate, center, and normalize embeddings.
    assert X_embeddings.shape[1] == Y_embeddings.shape[1], \
        'Embeddings must be of same dimension'

    total_num_samples = 15

    # Normalize embeddings.
    scaler = StandardScaler()
    scaler.fit(np.vstack((X_embeddings, Y_embeddings)))
    X_emb = scaler.transform(X_embeddings)
    Y_emb = scaler.transform(Y_embeddings)

    if distance_measure == persistence_distance:
        # compute diagrams first to reduce computation time and set consistent scale
        # compute D_xx
        HOMOLOGY_DIMENSION = 1
        start = time()
        X_sample_1_dgms = [
            _compute_persistence_diagram(
                subsample(X_emb, sample_n=sample_size),
                max_dimension = HOMOLOGY_DIMENSION
            )[HOMOLOGY_DIMENSION]
        for i in range(total_num_samples)]
        X_sample_2_dgms = [
            _compute_persistence_diagram(
                subsample(X_emb, sample_n=sample_size),
                max_dimension = HOMOLOGY_DIMENSION
            )[HOMOLOGY_DIMENSION]
         for i in range(total_num_samples)]
        stop = time()
        print(stop-start)
        # establish scale
        if HOMOLOGY_DIMENSION == 0:
            X_sample_1_dgms = [dgm[0][:-1] for dgm in X_sample_1_dgms]
            X_sample_2_dgms = [dgm[0][:-1] for dgm in X_sample_2_dgms]

        start = min([np.min(barcode) for dgm in X_sample_1_dgms for barcode in dgm])
        stop = max([np.max(barcode) for dgm in X_sample_1_dgms for barcode in dgm])

        pers_imager = PersistenceImager()
        pers_imager.fit(X_sample_1_dgms)
        features_0 = pers_imager.transform(X_sample_1_dgms, skew=True)
        features_1 = pers_imager.transform(X_sample_2_dgms, skew=True)

        D_xx_arr = [
            np.linalg.norm(feature_0 - feature_1)
        for feature_0, feature_1 in zip(features_0, features_1)]
 
        # compute D_xy
        X_sample_dgms = [
            _compute_persistence_diagram(
                subsample(X_emb, sample_n=sample_size),
                max_dimension = HOMOLOGY_DIMENSION
            )[HOMOLOGY_DIMENSION]
        for i in range(total_num_samples)]
        Y_sample_dgms = [
            _compute_persistence_diagram(
                subsample(Y_emb, sample_n=sample_size),
                max_dimension = HOMOLOGY_DIMENSION
            )[HOMOLOGY_DIMENSION]
         for i in range(total_num_samples)]

        if HOMOLOGY_DIMENSION == 0:
            X_sample_dgms = [dgm[0][:-1] for dgm in X_sample_dgms]
            Y_sample_dgms = [dgm[0][:-1] for dgm in Y_sample_dgms]

        features_X = pers_imager.transform(X_sample_dgms, skew=True)
        features_Y = pers_imager.transform(Y_sample_dgms, skew=True)

        D_xy_arr = [
            np.linalg.norm(feature_x - feature_y)
        for feature_x, feature_y in zip(features_X, features_Y)]

    else:
        D_xx_arr = []
        for i in range(total_num_samples):
            X_emb_sample_1 = subsample(X_emb, sample_n=sample_size)
            X_emb_sample_2 = subsample(X_emb, sample_n=sample_size)
            D_xx_arr.append(distance_measure(X_emb_sample_1, X_emb_sample_2))

        D_xy_arr = []
        for i in range(total_num_samples):
            X_emb_sample = subsample(X_emb, sample_n=sample_size)
            Y_emb_sample = subsample(Y_emb, sample_n=sample_size)
            D_xy_arr.append(distance_measure(X_emb_sample, Y_emb_sample))

    D_xx = np.median(D_xx_arr)
    D_xy = np.median(D_xy_arr)

    if decision == "stats":
        if len(D_xx_arr) <= 30 or len(D_xy_arr) <= 30:
            tstat, pvalue, _ = ws.ttest_ind(D_xx_arr, D_xy_arr, usevar="unequal")
        else:
            cm_obj = ws.CompareMeans(ws.DescrStatsW(D_xx_arr), ws.DescrStatsW(D_xy_arr))
            tstat, pvalue = cm_obj.ztest_ind(usevar="unequal")

        if pvalue < pvalue_threshold:
            decision_summary = "DECISION: Shift detected"
            decision_binary = True
        else:
            decision_summary = 'DECISION: No shift detected.'
            decision_binary = False
        decision_summary += f'\nD_xx: {D_xx:.4f}'
        decision_summary += f'\nD_xy: {D_xy:.4f}'
        decision_summary += f'\ntest statistic:  {tstat:.4f}'
        decision_summary += f'\np-value:  {pvalue:.4f}'
        decision_summary += f'\nTest decides "SIMILAR" if p-value <= {pvalue_threshold:.4f}.'
        return D_xx, D_xy, pvalue, decision_summary, decision_binary

    # TODO: Determine better ways to assign eps
    if eps == 0.0:
        if len(D_xx_arr) > 1:
            eps = 2 * np.std(D_xx_arr)
        else:
            eps = 0.0

    if D_xy - D_xx > eps:
        decision_summary = "DECISION: Shift detected"
        decision_binary = True
    else:
        decision_summary = 'DECISION: No shift detected.'
        decision_binary = False
    decision_summary += f'\nD_xx: {D_xx:.4f}'
    decision_summary += f'\nD_xy: {D_xy:.4f}'
    decision_summary += f'\neps:  {eps:.4f}'
    decision_summary += f'\nTest decides "SIMILAR" if D_xy - D_xx > eps.'

    return D_xx, D_xy, eps, decision_summary, decision_binary
