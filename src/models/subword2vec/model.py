import faiss
import numpy as np


class NNPredictor:
    def __init__(self, model, embeddings, tokens, normalize: bool = True):
        self.model = model
        self.tokens = tokens
        self.embeddings = embeddings
        self.normalize = normalize

        # L2 normalize embeddings
        if self.normalize:
            faiss.normalize_L2(embeddings)

        # Create token maps to translate between integer ids to tokens (and back)
        self.token_ids = np.arange(len(self.tokens))
        self.token_to_id_map = dict(zip(self.tokens, self.token_ids))
        self.id_to_token_map = dict(zip(self.token_ids, self.tokens))

        # Construct nn index
        self.faiss_index = self._create_faiss_index()

    def _create_faiss_index(self, links_per_vector: int = 64):
        faiss_index = faiss.IndexHNSWFlat(self.embeddings.shape[1],
                                          links_per_vector,
                                          faiss.METRIC_INNER_PRODUCT)
        faiss_index = faiss.IndexIDMap2(faiss_index)
        faiss_index.add_with_ids(self.embeddings, self.token_ids)

        return faiss_index

    def predict(self, token: str, k: int = 10):
        if token in self.tokens:
            token_id = int(self.token_to_id_map[token])
            token_embedding = self.faiss_index.reconstruct(token_id).reshape(1, -1)
        else:
            token_embedding = self.model.get_word_vector(token).reshape(1, -1)

        scores, nearest_neighbours_ids = self.faiss_index.search(token_embedding, k=k)
        nearest_neighbour_tokens = self._convert_ids_to_tokens(nearest_neighbours_ids)

        return nearest_neighbour_tokens, scores

    def _convert_ids_to_tokens(self, token_ids):
        return list(map(self.id_to_token_map.get, token_ids))