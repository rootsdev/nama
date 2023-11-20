from typing import Dict, List, Union

from mpire import WorkerPool
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import unidecode
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from src.data import constants


def top_similar_names(ref_vector, vectors_norm, names, threshold, top_n=20) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the top similar names and scores
    based on the cosine similarity between the ref vector and the list of vectors
    """
    # Normalize the reference vector and the list of vectors
    ref_vector_norm = ref_vector / np.linalg.norm(ref_vector)

    # Compute cosine similarity
    similarities = np.dot(vectors_norm, ref_vector_norm)

    # Filter based on the threshold
    above_threshold_indices = np.where(similarities > threshold)[0]

    if above_threshold_indices.size == 0:
        return np.array([]), np.array([])  # Return empty array if no vectors meet the threshold

    if above_threshold_indices.size > top_n:
        # Find indices of top_n similar vectors using argpartition
        partitioned_indices = np.argpartition(-similarities[above_threshold_indices], top_n - 1)[:top_n]
        top_indices = above_threshold_indices[partitioned_indices]
    else:
        top_indices = above_threshold_indices

    # Further sort the top_indices to order them by similarity
    top_indices_sorted = top_indices[np.argsort(-similarities[top_indices])]

    # Return the top similar vectors
    return names[top_indices_sorted], similarities[top_indices_sorted]


def _get_candidate_scores(shared, rows, _):
    source_names, source_names_X, num_candidates, metric, normalized = shared

    # get sorted score indexes - but partition first to save time
    if metric == "cosine":
        if normalized:  # If vectors are normalized dot product and cosine similarity are the same
            scores = safe_sparse_dot(rows, source_names_X.T)
        else:
            scores = cosine_similarity(rows, source_names_X)
        # partition and sort
        partitioned_idx = np.argpartition(scores, -num_candidates, axis=1)[:, -num_candidates:]
        sorted_idx = np.flip(np.argsort(np.take_along_axis(scores, partitioned_idx, axis=1), axis=1), axis=1)
    elif metric == "euclidean":
        scores = euclidean_distances(rows, source_names_X)
        # partition and sort
        partitioned_idx = np.argpartition(scores, num_candidates, axis=1)[:, :num_candidates]
        sorted_idx = np.argsort(np.take_along_axis(scores, partitioned_idx, axis=1), axis=1)
    else:
        raise ValueError("Unrecognized metric type. Valid options: 'cosine', 'euclidean'")
    # get the original score indexes before partitioning
    sorted_scores_idx = np.take_along_axis(partitioned_idx, sorted_idx, axis=1)

    # get sorted scores and source names
    sorted_scores = np.take_along_axis(scores, sorted_scores_idx, axis=1)
    sorted_source_names = source_names[sorted_scores_idx]

    return np.dstack((sorted_source_names.astype(object), sorted_scores))


def get_best_matches(
    input_names_X: np.ndarray,
    source_names_X: np.ndarray,
    source_names: np.ndarray,
    num_candidates: int = 10,
    metric: str = "cosine",
    normalized: bool = False,
    batch_size: int = 512,
    n_jobs: int = None,
    progress_bar: bool = True,
) -> np.ndarray:
    """
    A function that computes scores between the input names and the source names using the given metric type
    and returns a 3D array containing a list of best-matching (source-name, score) pairs for each input name

    :param input_names_X: Vectorized input names of shape (m, k) where m is the number of input names and
                          k is the dimensionality of each vectorized name.
    :param source_names_X: Vectorized source names of shape (n, k)
    :param source_names: an nd.array that contains the actual string value of source names
    :param num_candidates: Number of candidates to retrieve per name
    :param metric: Type of metric to use for fetching candidates
    :param normalized: Set it to true if X_input_names and X_source_names are L2 normalized
    :param batch_size: number of rows per batch
    :param n_jobs: set to the number of cpu's to use; defaults to all
    :param progress_bar: display progress bar
    :return: candidate_names_scores: an nd.array of [input names, candidates, (names, scores)]
    """
    batches = []
    for ix in range(0, input_names_X.shape[0], batch_size):
        # chunks needs to be a list of tuples; otherwise mpire passes each row in the chunk as a separate parameter
        batches.append((input_names_X[ix:ix + batch_size], ix))
    if n_jobs == 1:
        results = []
        if progress_bar:
            batches = tqdm(batches)
        for batch, ix in batches:
            results.append(_get_candidate_scores((source_names, source_names_X, num_candidates, metric, normalized), batch, ix))
        candidate_names_scores = np.vstack(results)
    else:
        with WorkerPool(shared_objects=(source_names, source_names_X, num_candidates, metric, normalized), n_jobs=n_jobs) as pool:
            candidate_names_scores = pool.map(_get_candidate_scores, batches, progress_bar=progress_bar)
    return candidate_names_scores


def get_euclidean_distance(names_sample: np.ndarray, names_encoded: np.ndarray, name1: str, name2: str) -> float:
    """
    Return the euclidean distance between name1 and name2
    :param names_sample: names in string form
    :param names_encoded: names in encoded form (in the same order)
    :param name1: specified name
    :param name2: specified name
    """
    enc1 = names_encoded[np.where(names_sample == name1)[0][0]]
    enc2 = names_encoded[np.where(names_sample == name2)[0][0]]
    return euclidean_distances([enc1], [enc2])[0][0]


def ndarray_to_exploded_df(candidates: np.ndarray, input_names: List[str], column_names: List[str]) -> pd.DataFrame:
    """
    Converts a 3d ndarray into an exploded pandas dataframe. Makes it easy to apply filters to the candidate set.
    :param candidates: Generated candidates for the given input names with shape (m, n, r)
    :param input_names: List of inputs names
    :param column_names: List of column names for the created dataframe
    :return: Pandas dataframe that has all the candidates in an exploded format
    """
    m, n, r = candidates.shape
    exploded_np = np.column_stack((np.repeat(input_names, n), candidates.reshape(m * n, -1)))
    exploded_df = pd.DataFrame(exploded_np, columns=column_names)
    return exploded_df


def convert_names_to_ids(names: np.ndarray, char_to_idx_map: Dict[str, int], max_len: int) -> np.array:
    """
    Convert a list of names to a 2d array of fixed-length integer sequences
    :param names: names to convert
    :param char_to_idx_map: character to id map
    :param max_len: maximum sequence length
    :return: 2d array of name ids
    """

    def convert_name(name):
        return [char_to_idx_map[c] for c in name]

    names_ids = list(map(convert_name, names))
    names_ids_chopped = [chop(name_ids, max_len) for name_ids in names_ids]
    names_ids_padded = [post_pad_to_length(name_ids, max_len) for name_ids in names_ids_chopped]
    return np.array(names_ids_padded)


def convert_ids_to_names(names_ids: np.ndarray, idx_to_char_map: Dict[int, str]) -> np.array:
    """
    Convert a 2d array of integer sequences to a 1d list of names
    :param names_ids: 2d array of name ids
    :param idx_to_char_map: id to character map
    :return: list of names
    """

    def convert_input_ids(input_ids):
        return "".join([idx_to_char_map[input_id] for input_id in input_ids])

    names = list(map(convert_input_ids, names_ids))
    return np.array(names)


def post_pad_to_length(input_ids: Union[list, np.ndarray], length: int) -> np.array:
    """
    Pad a sequence to a length
    :param input_ids: sequence (list or np.ndarray)
    :param length: length to pad
    :return: padded sequence
    """
    num_tokens = len(input_ids)
    if num_tokens < length:
        pad_width = length - num_tokens
        return np.pad(input_ids, (0, pad_width), "constant", constant_values=0)  # 0 is the pad id
    return np.array(input_ids)


def one_hot_encode(X: np.ndarray, vocab_length: int) -> np.ndarray:
    """
    One-hot encode values in X; e.g., if X is [0,1,2] and vocab_length is 3, then this will return [[1,0,0],[0,1,0],[0,0,1]]
    :param X: array to one-hot encode
    :param vocab_length: number of possible values in X
    :return: X that's been one-hot encoded
    """
    return np.eye(vocab_length)[X]


def chop(tokens: Union[list, np.ndarray], max_length: int) -> Union[list, np.ndarray]:
    """
    Chops tokens to have a maximum length
    :param tokens: tokens to chop (list or array)
    :param max_length: maximum length desired
    :return: chopped tokens
    """
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens


def build_token_idx_maps() -> (Dict[str, int], Dict[int, str]):
    """
    Return a map of character -> sequentially-increasing id, so a maps to 1, b maps to 2, etc.
    Also return the reverse map
    :return: map of character -> and id -> character
    """
    alphabet = list(constants.ALPHABET)
    idx = range(1, len(alphabet) + 1)
    char_to_idx_map = dict(zip(alphabet, idx))
    idx_to_char_map = dict(zip(idx, alphabet))

    char_to_idx_map[""] = 0
    idx_to_char_map[0] = ""

    return char_to_idx_map, idx_to_char_map


def remove_padding(name: str) -> str:
    """
    Remove begin and end tokens added by add_padding
    """
    return name[1:-1]


def add_padding(name: str) -> str:
    """
    Add BEGIN_TOKEN to the beginning of a name and END_TOKEN to the end of a name
    """
    return constants.BEGIN_TOKEN + name + constants.END_TOKEN


def check_convert_tensor(X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Ensure X is a torch tensor
    """
    if not torch.is_tensor(X):
        return torch.from_numpy(X)
    return X


def normalize(s: str) -> str:
    """
    Remove diacritics, lowercase, and strip
    """
    return unidecode.unidecode(s).lower().strip()


class ArrayDataLoader:
    """
    Data loader for chunking an array
    """

    def __init__(
        self,
        arr,
        batch_size,
        shuffle=False,
    ):
        self.arr = arr
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.arr)
        self.ix = 0
        return self

    def __next__(self):
        if self.ix >= self.arr.shape[0]:
            raise StopIteration
        # return a batch
        batch = self.arr[self.ix: self.ix + self.batch_size]
        self.ix += self.batch_size
        return batch
