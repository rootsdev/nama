import fasttext
import numpy as np


class SubwordModel:
    def __init__(self):
        self.model = None
        self.tokens = None
        self.embeddings = None

    def train(
        self,
        train_file_path: str,
        model: str = "skipgram",
        lr: float = 0.05,
        dim: int = 50,
        ws: int = 5,
        epoch: int = 10,
        neg: int = 5,
        minn: int = 3,
        maxn: int = 6,
        min_count: int = 1,
    ):
        self.model = fasttext.train_unsupervised(
            train_file_path,
            model=model,
            lr=lr,
            dim=dim,
            epoch=epoch,
            ws=ws,
            neg=neg,
            minn=minn,
            maxn=maxn,
            minCount=min_count,
        )

        # Skip first value since it's a special token
        self.tokens = np.array(self.model.words)[1:]
        self.embeddings = self.model.get_output_matrix()[1:]

    def get_embeddings(self, tokens: list[str]) -> np.ndarray:
        token_embeddings = []
        for token in tokens:
            token_embeddings.append(self.model.get_word_vector(token))
        return np.array(token_embeddings)
