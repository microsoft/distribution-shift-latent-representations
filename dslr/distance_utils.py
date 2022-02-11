# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Internal distance / similarity functions.
import numpy as np
import ot
from scipy.spatial.distance import cdist, pdist
from ripser import ripser

from dslr.faiss_utils import FAISSIndex
from dslr.persistence_utils import featurize_pointcloud


def energy_distance(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'linear'
) -> float:
    """Distances are euclidean.
    See: https://github.com/syrte/ndtest/blob/master/ndtest.py

    Args:
        x: First datset.
        y: Second dataset.
        method: Method of distance measurement.

    Returns:
        z: Energy distance value.
    """
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    # energy distance is 2*E[d(x,y)] - E[d(x,x*)] - E[d(y,y*)]
    z = (
        2.0 * dxy.sum() / (n * m) # 2*E[d(x,y)]
        - 2.0 * dx.sum() / n**2 # pdist returns distinct pairwise distances
        - 2.0 * dy.sum() / m**2 # pdist returns distinct pairwise distances
    )
    return z

def local_energy_distance(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    metric: str = 'euclidean',
    method: str = 'linear'
) -> float:
    """ Computes a local energy distance to reflect changes in geodesics

    Args:
        x: First dataset.
        y: Second dataset.
        k: Number of nearest neighbors to approximate geodesic.
        metric: Method to compute k nearest neighbor distances.
        method: Method of energy distance measurement.

    Returns:
        z: Energy distance value.
    """
    # Arg quality control check
    assert k < min(x.shape[0], y.shape[0]), \
        "k must be less than number of points in each of x and y."

    # Instantiate faiss index with X embeddings.
    E_x = FAISSIndex(
        embedding_dimension=x.shape[1],
        distance=metric
    )
    E_x.add_points(x)
    dx = E_x.index.search(x, k + 1)[0] # return knn distances within sample
    dyx = E_x.index.search(y, k + 1)[0] # return one sided knn distances across samples
    del E_x # clean up memory

    # Instantiate faiss index with Y embeddings.
    E_y = FAISSIndex(
        embedding_dimension=y.shape[1],
        distance=metric
    )
    E_y.add_points(y)
    dy = E_y.index.search(y, k + 1)[0] # return knn distances within sample
    dxy = E_y.index.search(x, k + 1)[0] # return one sided knn distances across samples
    del E_y # clean up memory

    # convert squared euclidean distances to euclidean distances
    if metric == 'euclidean':
        dx, dy, dxy, dyx = np.sqrt(dx), np.sqrt(dy), np.sqrt(dxy), np.sqrt(dyx)

    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy, dyx = np.log(dx), np.log(dy), np.log(dxy), np.log(dyx)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = ( # original energy distance is 2*E[d(x,y)] - E[d(x,x*)] - E[d(y,y*)]
            dxy.sum() / (k * n) + dyx.sum() / (k * m) # inter sample knn distances
            - dx.sum() / (k * n)
            - dy.sum() / (k * m) # intra sample knn distances
        )
    return z


def sliced_wasserstein_distance_persistence_features(
    x: np.ndarray,
    y: np.ndarray,
    dimension: int=0
) -> float:
    """Distances are euclidean.

    Args:
        x: First dataset.
        y: Second dataset.
        dimension: Dimension for persistence diagram to compute distance

    Returns:
        z: sliced wasserstein distance value.
    """
    dgm_0 = ripser(x, maxdim=dimension)['dgms'][dimension]
    dgm_1 = ripser(y, maxdim=dimension)['dgms'][dimension]

    if dimension == 0: # remove infinite bar
        dgm_0 = dgm_0[:-1]
        dgm_1 = dgm_1[:-1]

    return sliced_wasserstein_diagrams(dgm_0, dgm_1)


def persistence_distance(
    x: np.ndarray,
    y: np.ndarray,
    dimension: int=0,
    persistence_feature: str="persistence_landscape"
) -> float:
    """Distances are euclidean on persistence features.

    Args:
        x: First datset.
        y: Second dataset.
        dimension: Dimension for persistence diagram to compute distance.
        persistence_feature: Name of persistence featurizer.

    Returns:
        z: euclidean distance between persistence features.
    """
    return np.linalg.norm(
        featurize_pointcloud(x, dimension, persistence_feature) -
        featurize_pointcloud(y, dimension, persistence_feature)
    )
