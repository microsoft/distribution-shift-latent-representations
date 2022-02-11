# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pickle
import sys
from time import time

import numpy as np

from dslr.distance_utils import (
    energy_distance,
    local_energy_distance,
    sliced_wasserstein_distance_persistence_features,
)
from dslr.distribution_shift_utils import get_pairs, run


BASE = '../embeddings'

# # CASE: WILDS
# TEMPLATE = BASE + '/wilds/{}_featurizer_embeddings.pkl'
# DATASET_NAMES = ['ogb-molpcba', 'poverty', 'camelyon17', 'civilcomments']
# LOGFILE = 'run_experiments_wilds.log'
# SAMPLE_SIZE = 1000

# # CASE: MNIST
# TEMPLATE = BASE + '/mnist/{}_embeddings.pkl'
# DATASET_NAMES = ['mnist', 'mnist_sp', 'mnist_ds']
# LOGFILE = 'run_experiments_mnist.log'
# SAMPLE_SIZE = 1000

# CASE: MNIST_SMALL
TEMPLATE = BASE + '/mnist/{}_embeddings.pkl'
DATASET_NAMES = ['mnist_small', 'mnist_small_sp', 'mnist_small_ds']
LOGFILE = 'run_experiments_mnist_small.log'
SAMPLE_SIZE = 1000

TEST_NAMES = ['subsample']
DISTANCE_FN = local_energy_distance


# Load embedding data from blobstorage.
def load_data():
    data = {}
    for dataset_name in DATASET_NAMES:
        with open(TEMPLATE.format(dataset_name), 'rb') as f:
            data[dataset_name] = pickle.load(f)
        
        # Print shapes.
        for split_name in data[dataset_name].keys():
            print(
                dataset_name,
                split_name,
                data[dataset_name][split_name]['embeddings'].shape
            )

    return data

def main():
    data = load_data()
    pairs = get_pairs(data)
    print(pairs)

    # Optionally remove existing file.
    if os.path.exists(LOGFILE):
        os.remove(LOGFILE)

    run(data, pairs, LOGFILE, TEST_NAMES, SAMPLE_SIZE, DISTANCE_FN)

if __name__ == '__main__':
    main()
