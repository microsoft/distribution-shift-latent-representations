# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import random
from typing import Callable, List, Optional

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


class MNISTSampler(datasets.MNIST):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        num_total (int, optional): Maximum number of samples to sample from the original
            dataset.
        weights (List, optional): List of weights for each class (0-9). The data will be
            sampled based in these weights. All of the floats must be within the range
            [0,1].
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_total: int = -1,
            weights: List[float] = [1.0]*10
    ) -> None:
        super(MNISTSampler, self).__init__(root, train=train, transform=transform,
                                    target_transform=target_transform, download=download)
        
        assert max(weights) <= 1.0 and min(weights) >= 0.0, "Weights need to be in the range of [0,1]"
        
        if num_total == -1:
            num_total = len(self.data)
        
        # Aggregate indices for each class
        class_map = defaultdict(list)
        for i, target in enumerate(self.targets):
            class_map[target.item()].append(i)
            
        # Get indices per class based on weights
        new_indices = []
        for target_class in class_map.keys():
            target_indices = class_map[target_class]
            random.shuffle(target_indices)
            target_indices = target_indices[:int(weights[target_class]*len(target_indices))]
            new_indices.extend(target_indices)

        # Assign newly sampled data
        random.seed(42)
        random.shuffle(new_indices)
        self.new_indices = new_indices[:num_total]
        self.data = self.data[self.new_indices]
        self.targets = self.targets[self.new_indices]


def get_mnist_data(
        root,
        shift: str,
        size: str = "original",
        get_original_test: bool = False,
        download: bool = False
):
    """ Returns train and test mnist data for different shifts.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        shift (string): The type of shift to apply to the mnist dataset.
        size (string): Size of the dataset to return
        get_original_test: Returns the full test set if toggled
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    assert shift in ["subpopulation", "domain", "none"], "Shift type not recognized"
    assert size in ["original", "small"], "Size not recognized"

    train_num_total = -1
    test_num_total = -1
    train_weights = [1.0]*10
    test_weights = [1.0]*10

    if shift == "subpopulation":
        train_weights = [1]*5 + [0.1]*5
        test_weights = [0.1]*5 + [1]*5
    elif shift == "domain":
        train_weights = [1]*6 + [0]*4
        test_weights = [0]*6 + [1]*4

    if size == "small":
        train_num_total = 6000
        test_num_total = 1000

    if get_original_test:
        test_num_total = -1

    train_data = MNISTSampler(
        root = root,
        train = True,
        transform = ToTensor(),
        download = download,
        weights = train_weights,
        num_total = train_num_total
    )
    test_data = MNISTSampler(
        root = root,
        train = False,
        transform = ToTensor(),
        download = download,
        weights = test_weights,
        num_total = test_num_total
    )

    return train_data, test_data
