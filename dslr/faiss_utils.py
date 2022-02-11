# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
from typing import List, Tuple

import concurrent.futures
import faiss
import numpy as np
from bidict import bidict
from tqdm import tqdm


class FAISSIndex:
    """Initiates Faiss index and stores embedding space info.

    Stores name, model_name, embedding dimension, and point name to index
    dictionary.
    """

    def __init__(
        self,
        embedding_dimension: int,
        distance: str = 'euclidean',
        utilize_gpu: bool = False,
        normalized_index: bool = False
    ):
        supported_distance_metrics = ['euclidean', 'cosine-similarity']
        assert distance in supported_distance_metrics, \
            f'Distance metric must be one of: {supported_distance_metrics}'

        # Instantiate a new index.
        # Set flag to normalize incoming queries against the faiss index.
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.normalized_index = normalized_index 

        if distance == 'cosine-similarity':
            # Instantiate a new cosine simlarity index.
            self.construct_metric_index()
            self.normalized_index = True

        if utilize_gpu:
            self.convert_cpu_index_to_gpu_index()

    def construct_metric_index(self) -> None:
        """Constructs FAISS index whose distance is the L2 inner product."""
        self.index = faiss.IndexFlatIP(self.index.d)

    def convert_cpu_index_to_gpu_index(self) -> None:
        """Transfers an existing FAISS index to a gpu if available."""
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def add_points(
        self,
        embedding_arr: np.ndarray
    ) -> None:
        """
        Adds embeddings and updates point name and tag to the correct dictionaries.

        Args:
            embedding_arr: Array of embedding arrays.

        Returns:
            None
        """
        assert embedding_arr.shape[1] == self.index.d, \
            'Dimensions of embeddings and embedding space must match.'

        # Cast to float32.
        embedding_arr = embedding_arr.astype(np.float32)

        if self.normalized_index:
            norms = np.linalg.norm(embedding_arr, axis=1)[:, np.newaxis]
            embedding_arr = embedding_arr / norms
        self.index.add(embedding_arr)

    def remove_points_by_point_idx(self, point_idx_arr: np.ndarray) -> None:
        """Removes point info from dictionaries and embedding from index.

        Then updates all other indices to match the faiss index update.

        Args:
            point_idx_arr: Array of point indices.

        Returns:
            None
        """
        self.index.remove_ids(point_idx_arr)

    def extract_nearest_neighborhood_graph(
        self,
        embedding_arr: np.ndarray,
        num_nearest_neighbors: int,
        verbose: bool = False
    ) -> List:
        """Retrieves or passes embedding for specified point name unique identifiers.

        Args:
            embedding_arr: Array of embeddings to search with.
            num_nearest_neighbors: Number of neighbors to search with.
            verbose: Flag to include print statements.

        Returns:
            D: List of distances to matches.
            I: List of indices of matches.
        """
        if self.normalized_index:
            norms = np.linalg.norm(embedding_arr, axis=1)[:, np.newaxis]
            embedding_arr = embedding_arr / norms

        # Format data to faiss index requirement.
        query_arr = embedding_arr.astype(np.float32)

        if verbose:
            print('Querying faiss index.')
        start = time.time()
        # Represent nearest neighborhood graph as multiple arrays of source,
        # destination, and weight information.
        num_cpus = os.cpu_count() - 1
        num_threads = num_cpus * 2
        args = (
            (self, sub_arr, num_nearest_neighbors)
            for sub_arr in
            np.array_split(np.array(query_arr), num_threads)
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(
                executor.map(lambda p: query_faiss_index(*p), args)
            )

        if verbose:
            print('Time to query faiss index : {:.5f}'.format(time.time() - start))

        start = time.time()
        D = np.zeros(num_nearest_neighbors)
        I = np.zeros(num_nearest_neighbors).astype(int)
        for result in results:
            # Append distances and indices.
            D = np.vstack((D, result[0]))
            I = np.vstack((I, result[1]))
            # Append pass through indices.
        if verbose:
            print('Time to gather faiss results : {:.5f}'.format(
                time.time() - start))

        return D[1:], I[1:]


# The following helper functions use FAISSIndex, and exist to avoid circular
# references when running `import embedding_space`.

def query_faiss_index(
    faiss_index: FAISSIndex,
    query_embeddings_arr: np.ndarray,
    num_nearest_neighbors: int = 1
) -> List:
    """Facilitates concurrent.futures.ThreadPoolProcessor, by being unnested.

    Args:
        faiss_index: A FAISSIndex object.
        query_embeddings_arr: Array of query embeddings.
        num_nearest_neighbors: Number of nearest neighbors.

    Returns:
        distance_arr: List of distances to matches.
        faiss_idx_arr: List of indices of matches.
    """
    # Query nearest neighbors in index using the provided embeddings.
    # Does not require the embedding to be in the index.
    if len(query_embeddings_arr.shape) == 1:
        query_embeddings_arr = query_embeddings_arr.reshape(1, -1)
    distance_arr, faiss_idx_arr = faiss_index.index.search(
        query_embeddings_arr, num_nearest_neighbors)
    return distance_arr, faiss_idx_arr
