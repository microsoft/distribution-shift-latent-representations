# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np


def subsample(
    embeddings: np.ndarray,
    sample_percentage: float = None,
    sample_n: int = None
) -> np.ndarray:
    """Subsamples from array of embeddings.

    Note: If sample_n is >= embedding count, returns original embeddings.
    Note: If neither sample_percentage or sample_n is given, raises ValueError.

    Args:
        embeddings: Array of embeddings.
        sample_percentage: Percentage to keep.
        sample_n: Number to keep.

    Returns:
        sample_embeddings: Array of sampled embeddings.
    """
    # Case where sample size is too large. Returns original embeddings.
    if sample_n is not None and sample_n >= embeddings.shape[0]:
        return embeddings

    # Case where sample size is suitable, returns random n.
    elif sample_n is not None and sample_n > 0:
        return embeddings[np.random.choice(embeddings.shape[0], sample_n)]

    # Case where only percentage is given, return random k%.
    elif sample_percentage is not None:
        embeddings_size = len(embeddings)
        sample_size = int(embeddings_size * sample_percentage)
        sample_ids = np.random.choice(embeddings_size, sample_size)
        sample_embeddings = embeddings[sample_ids]

    else:
        raise ValueError('No sample_percentage or sample_n was found.')

    return sample_embeddings
