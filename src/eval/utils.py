import numpy as np
from typing import List, Tuple


def similar_names_scores_to_ndarray(similar_names_scores: List[List[Tuple]]) -> np.ndarray:
    # TODO how can I create a 3D array with (str, float) as the third axis without taking apart and re-assembling?
    # names is a 2D array [names, name of k similar-names]
    names = np.array(list(list(cell[0] for cell in row) for row in similar_names_scores), dtype="O")
    # scores is a 2D array [names, score of k similar-names]
    scores = np.array(list(list(cell[1] for cell in row) for row in similar_names_scores), dtype="f8")
    # similar_names_scores is now a 3D array [names, k similar-names, name or score]
    return np.dstack((names, scores))


class EvalDataLoader:
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
        batch = self.arr[self.ix : self.ix + self.batch_size]
        self.ix += self.batch_size
        return batch
