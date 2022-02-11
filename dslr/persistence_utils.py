# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from time import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from persim import bottleneck, PersImage, PersistenceImager, plot_diagrams
from persim import sliced_wasserstein as sliced_wasserstein_diagrams
from persim.landscapes import PersLandscapeApprox, vectorize
from ripser import ripser
from scipy.cluster import hierarchy


def _compute_persistence_diagram(
    embeddings: np.ndarray,
    max_dimension:int
) -> List:
    """Generates persistence diagrams for embeddings.

    Args:
        embeddings: Array of embeddings.
        max_dimension: specify maximum dimension of diagrams to compute 
        (lower dimensions are faster to compute).
    
    Returns:
        dgm_emb: List of persistence diagram results for embeddings.
    """
    return ripser(embeddings, maxdim=max_dimension)['dgms']


def featurize_pointcloud(
    embeddings: np.ndarray,
    dimension: int = 0,
    persistence_feature: str = "persistence_landscape"
):
    """Computes subsamples and featurizes the corresponding persistence diagrams.
    Args:
        embeddings: Array of embeddings, one embedding per row.
        dimension: Dimension for persistence diagram to compute distance
        persistence_feature: Name of featurization technique to apply to the input persistence diagram. Defaults to "persistence_landscape".

    Returns:
        persistence_feature: Embeddings of specified dimension persistence diagrams.
    """
    # compute persistence diagram for each sample
    # NOTE persistence computations are O(choose(subsample_size, dimension+2))**3) 
    dgm = _compute_persistence_diagram(
        embeddings = embeddings,
        max_dimension = dimension
    )[dimension]

    # remove infinite bars for dimension 0
    if dimension == 0:
        dgm = dgm[:-1]

    # create an embedding for each persistence diagram
    return persistence_featurizer(
        persistence_diagram = dgm,
        persistence_feature = persistence_feature
    )
