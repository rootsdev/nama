from typing import List

import fasttext
import numpy as np


class SubwordModel:
    def __init__(self):
        self.model = None
        self.tokens = None
        self.embeddings = None

    def train(self,
              train_file_path: str,
              model: str = 'skipgram',
              epoch: int = 10,
              ws: int = 5,
              neg: int = 5,
              min_count: int = 1):
        self.model = fasttext.train_unsupervised(train_file_path,
                                                 model=model,
                                                 epoch=epoch,
                                                 ws=ws,
                                                 neg=neg,
                                                 minCount=min_count)

        # Skip first value since it's a special token
        self.tokens = np.array(self.model.words)[1:]
        self.embeddings = self.model.get_output_matrix()[1:]

    def get_embeddings(self, tokens: List[str]) -> np.ndarray:
        token_embeddings = []
        for token in tokens:
            token_embeddings.append(self.model.get_word_vector(token))
        return np.array(token_embeddings)
