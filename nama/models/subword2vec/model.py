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
        faiss_index = faiss.IndexHNSWFlat(self.embeddings.shape[1], links_per_vector, faiss.METRIC_INNER_PRODUCT)
        faiss_index = faiss.IndexIDMap2(faiss_index)
        faiss_index.add_with_ids(self.embeddings, self.token_ids)

        return faiss_index

    def predict_batch(self, tokens: list[str], k: int = 10) -> tuple[list[list[str]], list[list[float]]]:
        # Get token embeddings
        token_embeddings = self.get_embeddings_batch(tokens)

        # Unit normalize embeddings
        if self.normalize:
            faiss.normalize_L2(token_embeddings)

        scores_batch, nearest_neighbours_ids_batch = self.faiss_index.search(token_embeddings, k=k)
        # Convert int ids to str tokens
        nearest_neighbour_tokens_batch = []
        for nn_ids in nearest_neighbours_ids_batch:
            nn_tokens = self._convert_ids_to_tokens(nn_ids.flatten().tolist())
            nearest_neighbour_tokens_batch.append(nn_tokens)

        return nearest_neighbour_tokens_batch, scores_batch

    def predict(self, token: str, k: int = 10) -> tuple[list[str], list[float]]:
        # Get token embedding
        token_embedding = self.get_embedding(token)

        # Reshape to match expected input shape for Faiss
        token_embedding = token_embedding.reshape(1, -1)

        # Unit normalize embedding
        if self.normalize:
            faiss.normalize_L2(token_embedding)

        scores, nearest_neighbours_ids = self.faiss_index.search(token_embedding, k=k)
        nearest_neighbour_tokens = self._convert_ids_to_tokens(nearest_neighbours_ids.flatten().tolist())

        return nearest_neighbour_tokens, scores.tolist()

    def get_embeddings_batch(self, tokens: list[str]):
        token_embeddings = []
        for token in tokens:
            token_embeddings.append(self.get_embedding(token))

        return np.array(token_embeddings)

    def get_embedding(self, token: str):
        if token in self.tokens:
            token_id = int(self.token_to_id_map[token])
            token_embedding = self.faiss_index.reconstruct(token_id)
        else:
            token_embedding = self.model.get_word_vector(token)

        return token_embedding

    def _convert_ids_to_tokens(self, token_ids: list[int]):
        return list(map(self.id_to_token_map.get, token_ids))
