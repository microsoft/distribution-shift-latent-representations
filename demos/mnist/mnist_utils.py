# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from mnist_data import MNISTSampler
from dslr.distance_utils import local_energy_distance, energy_distance
from dslr.distribution_shift_utils import run_approximate_shift_test, wrap_subsample_test
from dslr.persistence_utils import featurize_pointcloud
from sklearn import svm


# -

class CNN(nn.Module):
    """
    A simple model that consists of 2-layer CNN followed by 2-layer dense network
    used for the mnist classification task.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.out(x)
        return output, x


def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")


def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError(
            "Elements of the list to collate must be tensors or dicts.")


def train(num_epochs, model, loader, loss_func, optimizer):
    model.train()
    # Train the model
    total_step = len(loader)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for i, (images, labels) in enumerate(loader):
            output = model(images)[0]
            loss = loss_func(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += len(labels)
        accuracy = correct / total
        print('Test Accuracy of the model on the test images: %.2f' % accuracy)
    return accuracy


def extract_embeddings(model, loader):
    model.eval()
    data = {}
    with torch.no_grad():
        for split in loader.keys():
            embeddings = []
            all_labels = []
            for images, labels in loader[split]:
                test_output, last_layer = model(images)
                embeddings.append(detach_and_clone(last_layer))
                all_labels.extend(labels)
            embeddings = collate_list(embeddings)
            data[split] = {'embeddings': embeddings.cpu(), 'ids': all_labels}
    return data


def run_distribution_ablation(
    model,
    loss_func,
    optimizer_type,
    num_epochs = 2,
    num_samples = 3,
    sample_size = 1000,
    sampling_distribution = "random",
    distance_measure = energy_distance
):
    if sampling_distribution == "random":
        sample_weights_train = np.random.random([num_samples, 10])
        sample_weights_test = np.random.random([num_samples, 10])
    elif sampling_distribution == "dirichlet":
        alphas = np.logspace(-1.5, 4, num=num_samples)
        sample_weights_train = np.array([[np.random.dirichlet([alpha] * 10) for i in range(len(alphas))] for alpha in alphas]).reshape(-1,10)
        sample_weights_test = np.array([[np.random.dirichlet([alpha] * 10) for alpha in alphas] for i in range(len(alphas))]).reshape(-1,10)
    sample_weights_train /= np.amax(sample_weights_train, axis=1).reshape(-1,1)
    sample_weights_test /= np.amax(sample_weights_test, axis=1).reshape(-1,1)

    shift_test = wrap_subsample_test
    data_name = "mnist_ablation"

    accuracies = []
    embed_distances = []
    sample_distances = []
    proportions = []
    classification_accuracies = []
    shift_classification_scores = []
    for i in tqdm(range(len(sample_weights_train))):
        print('-'*100)
        print(f'Sample {i+1}/{len(sample_weights_train)}')
        curr_model = deepcopy(model)
        optimizer = optimizer_type(curr_model.parameters(), lr=0.001)
        print(sample_weights_train[i])
        print(sample_weights_test[i])
        train_data = MNISTSampler(
            root = 'data',
            train = True,
            transform = ToTensor(),
            weights = sample_weights_train[i],
        )
        test_data = MNISTSampler(
            root = 'data',
            train = False,
            transform = ToTensor(),
            weights = sample_weights_test[i],
        )
        loaders = {
            'train' : DataLoader(train_data,
                                 batch_size=100,
                                 shuffle=True,
                                 num_workers=1),

            'test'  : DataLoader(test_data,
                                 batch_size=100,
                                 shuffle=True,
                                 num_workers=1),
        }

        train(num_epochs, curr_model, loaders['train'], loss_func, optimizer)
        accuracy = test(curr_model, loaders['test'])
        accuracies.append(accuracy)
        embeddings = extract_embeddings(curr_model, loaders)
        embed_distances.append(distance_measure(np.array(embeddings['train']['embeddings']).astype(np.float32),
                                                np.array(embeddings['test']['embeddings']).astype(np.float32)
                                               )
                              )

        # Compute proportion of runs that detected a shift.
        config = {
            'dataset_name': data_name,
            'data': {data_name: {'train': {'embeddings': np.array(embeddings['train']['embeddings']).astype(np.float32)},
                                 'test': {'embeddings': np.array(embeddings['test']['embeddings']).astype(np.float32)}
                                }
                    },
            'pair': ('train', 'test'),
            'test_name': 'subsample',
            'shift_test': shift_test,
            'distance_measure': distance_measure,
            'sample_size': sample_size,
            'logfile': 'ablation_subsample.log'
        }
        decision_counts, runs_dxx, runs_dxy = run_approximate_shift_test(config)
        # Calculate shift classification score
        dxx_mean = np.mean(runs_dxx)
        dxx_std = np.std(runs_dxx)
        dxx_correct = sum([1 if dxx-dxx_mean-2*dxx_std < 0 else 0 for dxx in runs_dxx])
        dxy_correct = sum([1 if dxy-dxx_mean-2*dxx_std >= 0 else 0 for dxy in runs_dxy])
        shift_classification_score = (dxy_correct + dxx_correct)/(len(runs_dxx) + len(runs_dxy))

        proportion_detection = decision_counts[True] / sum(decision_counts.values())
        classification_accuracy, _ = subsample_classification_test(
            np.array(embeddings['train']['embeddings']).astype(np.float32),
            np.array(embeddings['test']['embeddings']).astype(np.float32),
            featurizer = featurize_pointcloud,
            classifier=svm.SVC(),
            num_subsamples = sum(decision_counts.values()),
            subsample_size = sample_size,
        )
        classification_accuracies.append(classification_accuracy)
        shift_classification_scores.append(shift_classification_score)
        proportions.append(proportion_detection)

        sample_distances.append(
            np.linalg.norm(sample_weights_test[i] - sample_weights_train[i])
        )

    return accuracies, embed_distances, sample_distances, proportions, classification_accuracies, shift_classification_scores, sample_weights_train, sample_weights_test
