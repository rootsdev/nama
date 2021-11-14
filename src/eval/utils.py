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


