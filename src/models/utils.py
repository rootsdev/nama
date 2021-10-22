from typing import Dict, List, Union, Set

import heapq
import jellyfish
import pandas as pd
import numpy as np
import torch
import unidecode
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from src.data import constants


def get_best_matches(
    input_names_X: np.ndarray,
    source_names_X: np.ndarray,
    source_names: np.ndarray,
    num_candidates: int = 10,
    metric: str = "cosine",
    normalized: bool = False,
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
    :return: candidate_names_scores: an nd.array of shape (number of input names, num_candidates, Z)
            (Z=0 has names, Z=1 has scores)
    """
    step_size = 1000
    candidate_names_scores = []
    for ix in range(0, input_names_X.shape[0], step_size):
        rows = input_names_X[ix : ix + step_size]
        sorted_scores_idx = None

        if metric == "cosine":
            if normalized:  # If vectors are normalized dot product and cosine similarity are the same
                scores = safe_sparse_dot(rows, source_names_X.T)
            else:
                scores = cosine_similarity(rows, source_names_X)
        elif metric == "euclidean":
            scores = euclidean_distances(rows, source_names_X)
            sorted_scores_idx = np.argsort(scores, axis=1)[:, :num_candidates]
        else:
            raise ValueError("Unrecognized metric type. Valid options: 'cosine', 'euclidean'")

        if sorted_scores_idx is None:
            sorted_scores_idx = np.flip(np.argsort(scores, axis=1), axis=1)[:, :num_candidates]

        sorted_scores = np.take_along_axis(scores, sorted_scores_idx, axis=1)
        ranked_candidates = source_names[sorted_scores_idx]
        candidate_names_scores.append(np.dstack((ranked_candidates, sorted_scores)))
    return np.vstack(candidate_names_scores)


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
    name_ids_chopped = [chop(name_id, max_len) for name_id in names_ids]
    name_ids_padded = [post_pad_to_length(name_id, max_len) for name_id in name_ids_chopped]
    return np.array(name_ids_padded)


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
        return np.pad(input_ids, (0, pad_width), "constant", constant_values=0)
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


def convert_names_to_model_inputs(
    names: Union[list, np.ndarray], char_to_idx_map: dict, max_name_length: int
) -> (torch.Tensor, torch.Tensor):
    """
    Return a torch tensor of names, where each name has been converted to a sequence of ids and the ids have been one-hot encoded.
    Also return the tensor where the names of ahve converted to a sequence of ids but before the ids have been one-hot encoded.
    :param names: list of names to encode
    :param char_to_idx_map: map characters to ids
    :param max_name_length: maximum name length
    :return: 3D tensor, where axis 0 is names, axis 1 is character position and axis 2 is the one-hot encoding (a..z).
             also a 2D tensor of names->ids
    """
    X_targets = convert_names_to_ids(names, char_to_idx_map, max_name_length)
    X_one_hot = one_hot_encode(X_targets, constants.VOCAB_SIZE + 1)

    X_inputs = check_convert_tensor(X_one_hot)
    X_targets = check_convert_tensor(X_targets)

    return X_inputs, X_targets


def check_convert_tensor(X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Ensure X is a torch tensor
    """
    if not torch.is_tensor(X):
        return torch.from_numpy(X)
    return X


def get_k_near_negatives(name: str, positive_names: Set[str], all_names: Set[str], k: int) -> List[str]:
    """
    Return the k names from all_names that are most-similar to the input name and are not in positive_names
    :param name: input name
    :param positive_names: names that are matches to the specified name
    :param all_names: all candidate names
    :param k: how many names to return
    :return: list of names
    """
    # TODO re-think this function?
    similarities = {}
    for cand_name in all_names:
        if cand_name != name and cand_name not in positive_names:
            dist = jellyfish.levenshtein_distance(name, cand_name)
            similarity = 1 - (dist / max(len(name), len(cand_name)))
            similarities[cand_name] = similarity
    return heapq.nlargest(k, similarities.keys(), lambda n: similarities[n])


def normalize(s: str) -> str:
    """
    Remove diacritics, lowercase, and strip
    """
    return unidecode.unidecode(s).lower().strip()
