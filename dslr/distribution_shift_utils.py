# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from time import time
from tqdm import tqdm
from typing import Callable, List, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from dslr.comparisons import (
    perturbation_tolerance_test,
    subsample_based_distribution_shift_test,
    plot_perturbation_sensitivity,
)
from dslr.distance_utils import energy_distance, local_energy_distance
from dslr.sampling_utils import subsample


def wrap_subsample_test(
    emb_0: np.ndarray,
    emb_1: np.ndarray,
    config: dict,
    verbose: bool = False
) -> Tuple[float, float, float, str, bool]:
    """Wraps Subsample Test, for easy computation with only embedding arrays.
    """
    dataset_name = config['dataset_name']
    test_name = config['test_name']
    pair = config['pair']
    logfile = config['logfile']
    distance_measure = config['distance_measure']
    sample_size = config['sample_size']

    NUM_EPOCHS = 1

    t0 = time()

    # Run the test.
    dxx, dxy, eps, decision, decision_binary = subsample_based_distribution_shift_test(
        np.array(emb_0).astype(np.float32),
        np.array(emb_1).astype(np.float32),
        distance_measure,
        eps=0.0,
        num_epochs=NUM_EPOCHS,
        sample_size=sample_size, 
        decision='stats',
        pvalue_threshold=0.05
    )

    time_single_run = time() - t0
    if verbose:
        print(f'Time (subsample test): {time_single_run:.2f} s')

    # Write output to log.
    with open(logfile, 'a+') as f:
        _decision = decision.replace("\n", "; ")
        _decision = _decision.replace(",", " ")  # Replace comma so result is properly comma-separable
        f.write(f'{dataset_name},{" ".join(pair)},subsample,{dxx},{dxy},{eps},{_decision},{time_single_run}\n')
    if verbose:
        print(decision)

    return dxx, dxy, eps, decision, decision_binary


def wrap_perturbation_test(
    emb_0: np.ndarray,
    emb_1: np.ndarray,
    config: dict,
    verbose: bool = False
) -> Tuple[dict, float, float, float, str, bool]:
    """Wraps Perturbation Test, for easy computation with only embedding arrays.
    """
    dataset_name = config['dataset_name']
    test_name = config['test_name']
    pair = config['pair']
    logfile = config['logfile']
    distance_measure = config['distance_measure']
    sample_size = config['sample_size']

    t0 = time()

    # Run the test.
    sensitivity, dxy, d_star, c_star, decision, decision_binary = perturbation_tolerance_test(
        np.array(emb_0).astype(np.float32),
        np.array(emb_1).astype(np.float32),
        distance_measure,
        criteria='knn_recall',
        criteria_threshold=0.80,
        sample_size=sample_size
    )

    time_single_run = time() - t0

    if verbose:
        fig = plot_perturbation_sensitivity(
            sensitivity,
            distance_star=d_star,
            criteria_star=c_star,
            tag=f'{dataset_name} {pair}'
        )

        print(f'Time (perturbation test): {time_single_run:.2f} s')

    # Write output to log.
    with open(logfile, 'a+') as f:
        _decision = decision.replace("\n", "; ")
        _decision = _decision.replace(",", " ")  # Replace comma so result is properly comma-separable
        # NOTE: Logged output does not include sensitivity points.
        f.write(f'{dataset_name},{" ".join(pair)},perturbation,{dxy},{d_star},{c_star},{_decision},{time_single_run}\n')
    if verbose:
        print(decision)
        fig.show()

    return sensitivity, dxy, d_star, c_star, decision, decision_binary


def get_pairs(data: dict) -> dict:
    """Gets list of pairs of splits to run.
    E.g. Given splits ['train', 'val', 'id_test', 'od_test'], produces list:
         [
           ['train', 'id_test'],
           ['train', 'od_test'],
           ['id_test', 'od_test']
         ]

    Args:
        data: Dict, where top-level keys are dataset names, and next level keys
            are split names.

    Returns:
        pairs: Dict with dataset name as key, and list of split pairs as value.
    """
    pairs = {}
    for dataset_name in data.keys():
        split_names = data[dataset_name].keys()
        splits = [n for n in split_names if (('train' in n) or ('test' in n))]
        pairs[dataset_name] = list(itertools.combinations(splits, 2))

        print(dataset_name, pairs[dataset_name])
    return pairs


def run_approximate_shift_test(
    config: dict,
    verbose: bool = False
) -> bool:
    """
    Due to computational limitations, we can not run the shift test on the
    whole set of embeddings. Here, we run an approximation using the following
    approach.

    1. Subsample the pairs of data given the sample size
    2. Run the shift test num_runs times
    3. Aggregate the results from all the outputs

    Args:
        config: dictionary containing all the parameters required. To name a
            few, it includes data, dataset_name, pair, sample_size, and
            shift_test.
        verbose: boolean flag to toggle more verbose output.

    Return:
        A boolean decision is returned indicating if the distributions are
        similar or not.
    """
    data = config['data']
    dataset_name = config['dataset_name']
    pair = config['pair']
    sample_size = config['sample_size']
    shift_test = config['shift_test']

    if 'num_runs' not in config.keys():
        count_0 = data[dataset_name][pair[0]]['embeddings'].shape[0]
        count_1 = data[dataset_name][pair[1]]['embeddings'].shape[0]
        num_runs_0 = np.ceil(count_0 / sample_size)
        num_runs_1 = np.ceil(count_1 / sample_size)
        num_runs = int(max(num_runs_0, num_runs_1))
    else:
        num_runs = config['num_runs']

    # Collect results from all runs.
    runs_dxx, runs_dxy, runs_pvals = [], [], []

    decision_counts = {True:0, False:0}

    for i in range(num_runs):
        if verbose:
            print(f'Run {i+1}/{num_runs}')
        out_tuple = shift_test(
            data[dataset_name][pair[0]]['embeddings'],
            data[dataset_name][pair[1]]['embeddings'],
            config=config,
            verbose=verbose
        )

        # Based on the test, extract dxx and dxy results.
        if config['test_name'] == 'perturbation':
            runs_dxx.append(out_tuple[2])
            runs_dxy.append(out_tuple[1])
        elif config['test_name'] == 'subsample':
            runs_dxx.append(out_tuple[0])
            runs_dxy.append(out_tuple[1])
            runs_pvals.append(out_tuple[2])

        decision_binary = out_tuple[-1]
        decision_counts[decision_binary] += 1

    return decision_counts, runs_dxx, runs_dxy, runs_pvals


def run(
    data: dict,
    pairs: dict,
    logfile: str,
    test_names: List[str],
    sample_size: int,
    distance_measure: Callable = energy_distance
) -> None:
    """Runs panel of experiments.

    Args:
        data: Dict as follows:
            {
                'dataset_name': {
                    'split_name': {
                        'embeddings': <np.ndarray>,
                        'ids': List
                    },
                    ...
                },
                ...
            }
        pairs: Dict as follows:
            {'dataset_name': [('train', 'test'), ('train', 'in_domain_test')]}
        logfile: Path to text file for logging results.
        test_names: List of test names to run.
        sample_size: Size of subsamples to use from splits.
        distance_measure: Distribution distance function between sets of embeddings.

    Returns:
        None
    """
    start = time()

    print(
        'RUN CONFIG:\n',
        f'Dataset names: {list(data.keys())}\n',
        f'Log file: {logfile}\n',
        f'Test names: {test_names}\n',
        f'Sample size: {sample_size}'
    )

    available_tests = {
        'subsample': wrap_subsample_test,
        'perturbation': wrap_perturbation_test,
    }

    # Count number of results expected, for comparison at the end.
    num_results_expected = 0

    # Run full panel of experiments.
    for dataset_name in data:
        for test_name in test_names:

            # Select distribution shift function to use.
            shift_test = available_tests[test_name]

            config = {
                'dataset_name': dataset_name,
                'test_name': test_name,
                'logfile': logfile,
                'distance_measure': distance_measure,
                'data': data,
                'sample_size': sample_size,
                'shift_test': shift_test
            }

            # For perturbation test, number of subsamples is determined within test.
            if test_name == 'perturbation':
                config['num_runs'] = 1
            # For subsample test, fix number of runs for subsample test, which runs the minimum (2) subsamples.
            # TODO: Uncomment for ablations.
            elif test_name  == 'subsample':
                config['num_runs'] = 20

            # Run for each pair of splits.
            pairs_with_baseline = [('train', 'train')] + pairs[dataset_name]
            for pair in pairs_with_baseline:
                # e.g. pair = ['train', 'id_test']
                config['pair'] = pair

                print(
                    f'\n\nRUNNING > Dataset: {dataset_name.upper()}, '
                    f'Shift: {test_name.upper()}, Pair: {pair}'
                )

                # Get and report results from all runs.
                (
                    decision_counts,
                    runs_dxx,
                    runs_dxy,
                    run_pvals,
                ) = run_approximate_shift_test(config)
                print(f'Shift detected: {decision_counts}')
                print(f'Dxx (5%, 95%): {np.percentile(runs_dxx, [5, 95])}')
                print(f'Dxy (5%, 95%): {np.percentile(runs_dxy, [5, 95])}')
                dxx_mean = np.mean(runs_dxx)
                dxx_std = np.std(runs_dxx)
                dxx_correct = sum([1 if dxx < dxx_mean + 2 * dxx_std else 0 for dxx in runs_dxx])
                dxy_correct = sum([1 if dxy >= dxx_mean + 2 * dxx_std else 0 for dxy in runs_dxy])
                shift_classification_score = (dxy_correct + dxx_correct)/(len(runs_dxx) + len(runs_dxy))
                print(f'Shift classification score: {shift_classification_score}')

            num_results_expected += len(pairs_with_baseline) * sum(decision_counts.values())

    print(f'\n\nTotal run time (s): {time() - start:.2f}')
    print(f'Logfile: {logfile}')
    with open(logfile) as f:
       num_results = sum(1 for _ in f)
    print(f'Expected {num_results_expected} results, logged {num_results}.')


def analyze_local_energy_metric(
    data: dict,
    pairs: dict,
    tag: str = None,
    num_dist_samples: int = 50,
    subsample_size: int = 500
) -> plt.Figure:
    """Show performance of local_energy_distance versus energy_distance.

    Args:
        data: Dict as follows:
            {
                'dataset_name': {
                    'split_name': {
                        'embeddings': <np.ndarray>,
                        'ids': List
                    },
                    ...
                },
                ...
            }
        pairs: Dict as follows:
            {'dataset_name': [('train', 'test'), ('train', 'in_domain_test'), ...]}
        tag: String to uniquely identify saved matplotlib figure.
        num_dist_samples: Number of times to run each of energy and local energy.
        subsample_size: Size of each subsample sent to energy and local energy.

    Returns:
        fig: Summary figure of results.
    """
    colors = ['blue', 'green', 'black', 'yellow', 'brown']
    markers = ['3', 'o', '*', 'v', '+']
    assert (
        False not in [len(pairs_per_dataset) <= len(colors) for pairs_per_dataset in pairs.values()]
    ), 'Need as many colors as split pairs, for each dataset.'
    assert (
        False not in [len(pairs_per_dataset) <= len(markers) for pairs_per_dataset in pairs.values()]
    ), 'Need as many markers as split pairs, for each dataset.'

    print(
        'RUN CONFIG:\n'
        f'Dataset names: {list(data.keys())}\n'
        f'Num distance samples: {num_dist_samples}\n'
        f'Subsample size: {subsample_size}\n'
    )

    fig_dim = 5
    fig, axes = plt.subplots(1, len(data), figsize=(fig_dim * len(data), fig_dim))

    for i, dataset_name in tqdm(enumerate(data)):
        #pairs_with_baseline = [('train', 'train')] + pairs[dataset_name]

        # Ensure that ('train', 'test') is first among pairs.
        _pairs = pairs[dataset_name]
        if ('train', 'test') in _pairs:
            _pairs.insert(0, _pairs.pop(_pairs.index(('train', 'test'))))

        pairs_with_baseline = [('train', 'train')] + _pairs

        for j, pair in enumerate(pairs_with_baseline):
            d1 = np.array(data[dataset_name][pair[0]]['embeddings']).astype(np.float32)
            d2 = np.array(data[dataset_name][pair[1]]['embeddings']).astype(np.float32)
            pair_label = '-'.join(pair)

            # Several times, subsample and compute distances.
            sub_energy = []
            sub_local_energy = []
            for _ in range(num_dist_samples):
                sub1 = subsample(d1, sample_n=subsample_size)
                sub2 = subsample(d2, sample_n=subsample_size)

                # Compute energy and local energy.
                sub_energy.append(energy_distance(sub1, sub2))
                sub_local_energy.append(local_energy_distance(sub1, sub2))

            # Plot results, colored by split pair.
            axes[i].scatter(
                sub_energy,
                sub_local_energy,
                c=colors[j],
                marker=markers[j],
                label=pair_label,
                alpha=0.3
            )

        axes[i].legend()
        axes[i].set_title(f'{dataset_name}')
        axes[i].set_xlabel('Energy Distance')
        axes[i].set_ylabel('Local Energy Distance')

    title = (
        'Energy vs Local Energy, '
        f'Num samples: {num_dist_samples}, '
        f'Subsample size: {subsample_size}'
    )
    fig.suptitle(title, fontsize=14)

    if tag is not None:
        save_file = f'local_energy_analysis_{tag}_numsamp{num_dist_samples}_subsampsize{subsample_size}.png'
    else:
        save_file = f'local_energy_analysis_numsamp{num_dist_samples}_subsampsize{subsample_size}.png'

    plt.savefig(save_file)
    print(f'Saved figure to {save_file}')

    return fig
