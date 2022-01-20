import numpy as np
from typing import List, Tuple


def similars_to_ndarray(similar_names_scores: List[List[Tuple]]) -> np.ndarray:
    """
    Convert a list of list of (name, score) to a 3D array [names, similar-names, (name, score)]
    """
    # TODO how can I create a 3D array with (str, float) as the third axis without taking apart and re-assembling?
    # names is a 2D array [names, name of k similar-names]
    names = np.array(list(list(cell[0] for cell in row) for row in similar_names_scores), dtype="O")
    # scores is a 2D array [names, score of k similar-names]
    scores = np.array(list(list(cell[1] for cell in row) for row in similar_names_scores), dtype="f8")
    # similar_names_scores is now a 3D array [names, k similar-names, (name, score)]
    return np.dstack((names, scores))


def levenshtein(seqs1, seqs2):
    """
    This is an order of magnitude slower than jellyfish, so don't use this without thinking about how to make it faster
    """
    seq_lens1 = [len(seq) for seq in seqs1]
    seq_lens2 = [len(seq) for seq in seqs2]
    max_len = max(*seq_lens1, *seq_lens2)
    seqs1 = np.array([[ord(seq[i]) if i < len(seq) else 0 for i in range(max_len)] for seq in seqs1])
    seqs2 = np.array([[ord(seq[i]) if i < len(seq) else 0 for i in range(max_len)] for seq in seqs2])
    matrix = np.zeros((max_len+1, max_len+1, len(seqs1), len(seqs2)), dtype=int)
    for i in range(max_len+1):
        matrix [i, 0] = i
        matrix [0, i] = i

    cost = (np.subtract.outer(seqs1, seqs2) != 0).astype(int)
    cost = cost.transpose((1, 3, 0, 2))  # re-arrange so (pos_x, pos_y, seq_i, seq_j)

    for x in range(1, max_len+1):
        for y in range(1, max_len+1):
            matrix[x, y] = np.minimum(np.minimum(
                matrix[x-1, y] + 1,
                matrix[x-1, y-1] + cost[x-1, y-1]),
                matrix[x, y-1] + 1
            )

    result = np.zeros((len(seqs1), len(seqs2)), dtype=int)
    for i in range(len(seqs1)):
        for j in range(len(seqs2)):
            result[i, j] = matrix[seq_lens1[i], seq_lens2[j], i, j]

    return result
