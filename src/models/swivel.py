from collections import Counter
import math
import random

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models.utils import get_best_matches


class SwivelDataset:

    def __init__(self, input_names, weighted_actual_names, vocab_size=200000, symmetric=True):

        # get the most-frequent names from weighted_actual_names
        name_counter = Counter()
        for input_name, wan in zip(input_names, weighted_actual_names):
            for name, _, co_occurrence in wan:
                name_counter[input_name] += co_occurrence
                name_counter[name] += co_occurrence
        self._word2id = {w: i for i, (w, _) in enumerate(name_counter.most_common(vocab_size))}
        self._id2word = {i: w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)

        # create co-occurrence matrix as a sparse matrix
        dok = dok_matrix((self._vocab_len, self._vocab_len), dtype=np.int32)
        for input_name, wan in zip(input_names, weighted_actual_names):
            if input_name not in self._word2id:
                continue
            input_id = self._word2id[input_name]
            for name, _, co_occurrence in wan:
                if name not in self._word2id:
                    continue
                if co_occurrence == 0:
                    continue
                name_id = self._word2id[name]
                dok[name_id, input_id] += co_occurrence
                dok[input_id, input_id] += co_occurrence
                if symmetric:
                    dok[input_id, name_id] += co_occurrence
                    dok[name_id, name_id] += co_occurrence

        self._sparse_cooc = csr_matrix(dok)

        # create marginals
        row_sums = self.get_row_sums()
        col_sums = self.get_col_sums()

        # create submatrix row and column indices
        self._row_ixs = np.argsort(-row_sums)
        self._col_ixs = np.argsort(-col_sums)

        # calculate matrix log sum
        self._matrix_log_sum = math.log(row_sums.sum() + 1)

        print("Vocabulary length: {}".format(self._vocab_len))

    def get_vocab(self):
        return self._word2id

    def get_matrix_log_sum(self):
        return self._matrix_log_sum

    def get_num_rows(self):
        return len(self._row_ixs)

    def get_num_cols(self):
        return len(self._col_ixs)

    def get_row_sums(self) -> torch.tensor:
        return torch.tensor(np.asarray(self._sparse_cooc.sum(axis=1)).squeeze(), dtype=torch.int32)

    def get_col_sums(self) -> torch.tensor:
        return torch.tensor(np.asarray(self._sparse_cooc.sum(axis=0)).squeeze(), dtype=torch.int32)

    def get_submatrix_indices(self, row_ix, col_ix, n_shards_row, n_shards_col) -> (torch.tensor, torch.tensor):
        """
        Get the submatrix at row_ix, col_ix
        :param row_ix: row shard number
        :param col_ix: col shard number
        :param n_shards_row: number of shards in the rows axis
        :param n_shards_col: number of shards in the columns axis
        :return: (row_ixs, col_ixs)
        """
        return self._row_ixs[row_ix::n_shards_row], self._col_ixs[col_ix::n_shards_col]

    def get_submatrix_co_occurrences(self, row_indices, col_indices) -> torch.tensor:
        return torch.tensor(self._sparse_cooc[row_indices][:, col_indices].todense(), dtype=torch.int32)


class SwivelModel(nn.Module):
    """
    Adapted from https://github.com/src-d/tensorflow-swivel/blob/master/swivel.py
    """
    def __init__(self, num_embeddings, embedding_dim, confidence_base=0.1, confidence_scale=0.25, confidence_exponent=0.5):
        super().__init__()
        self.confidence_base = confidence_base
        self.confidence_scale = confidence_scale
        self.confidence_exponent = confidence_exponent

        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        nn.init.xavier_normal_(self.wi.weight.data)
        nn.init.xavier_normal_(self.wj.weight.data)
        nn.init.xavier_normal_(self.bi.weight.data)
        nn.init.xavier_normal_(self.bj.weight.data)

    def init_params(self, row_sums, col_sums, embeddings=None):
        # initialize weights with initial embeddings if passed in
        if embeddings is not None:
            self.wi.weight.data = torch.tensor(embeddings, dtype=torch.float32)
            self.wj.weight.data = torch.tensor(embeddings, dtype=torch.float32)
        # initialize biases with log of sums
        self.bi.weight.data = torch.unsqueeze(torch.tensor(list(math.log(x+1) for x in row_sums), dtype=torch.float32), 1)
        self.bj.weight.data = torch.unsqueeze(torch.tensor(list(math.log(x+1) for x in col_sums), dtype=torch.float32), 1)

    def forward(self, row_indices: torch.tensor, col_indices: torch.tensor, co_occurrences: torch.tensor, matrix_log_sum: float):
        w_i = self.wi(row_indices)
        w_j = self.wj(col_indices)
        b_i = self.bi(row_indices)
        b_j = self.bj(col_indices)

        # print("row_indices", row_indices.shape)
        # print("col_indices", col_indices.shape)
        # print("w_i", w_i.shape)
        # print("w_j", w_j.shape)
        # print("b_i", b_i.shape)
        # print("b_j", b_j.shape)
        # print("co_occurrences", co_occurrences.shape)

        # Multiply the row and column embeddings to generate predictions
        predictions = torch.matmul(w_i, w_j.T)
        # print("predictions", predictions.shape)

        # These binary masks separate zero from non-zero values
        count_is_zero = (co_occurrences == 0).type(torch.int8)
        count_is_nonzero = 1 - count_is_zero
        # print("count is zero", count_is_zero.shape)
        # print("count is nonzero", count_is_nonzero.shape)

        objectives = torch.log(co_occurrences + 1e-30) * count_is_nonzero
        objectives -= b_i
        objectives -= b_j.squeeze()
        objectives += matrix_log_sum
        # print("objectives", objectives.shape)

        err = predictions - objectives
        # print("err", err.shape)

        # The confidence function scales the L2 loss based on the raw co-occurrence count
        l2_confidence = self.confidence_base + self.confidence_scale * torch.pow(co_occurrences, self.confidence_exponent)
        # print("l2_confidence", l2_confidence.shape)
        loss_multiplier = 1 / math.sqrt(len(row_indices) * len(col_indices))

        l2_loss = loss_multiplier * torch.sum(0.5 * l2_confidence * err * err * count_is_nonzero)
        # print("l2_loss", l2_loss.shape)
        sigmoid_loss = loss_multiplier * torch.sum(F.softplus(err) * count_is_zero)
        # print("sigmoid_loss", sigmoid_loss.shape)
        return l2_loss + sigmoid_loss


def train_swivel(model, dataset, n_steps=100, submatrix_size=1024, lr=0.05, device="cpu", optimizer=None, verbose=True):
    model = model.to(device)
    if optimizer is None:
        optimizer = optim.Adagrad(model.parameters(), lr=lr)

    n_shards_row = math.ceil(dataset.get_num_rows() / submatrix_size)
    n_shards_col = math.ceil(dataset.get_num_cols() / submatrix_size)
    matrix_log_sum = dataset.get_matrix_log_sum()
    loss_values = list()

    if n_steps > 0:
        # get a random sequence of shards
        shard_positions = [(random.randrange(0, n_shards_row), random.randrange(0, n_shards_col))
                           for _ in range(0, n_steps)]
    else:
        # visit each shard once in random order
        shard_positions = [(row, col) for row in range(0, n_shards_row) for col in range(0, n_shards_col)]
        shard_positions = random.sample(shard_positions, len(shard_positions))

    for step, shard_position in enumerate(shard_positions):
        shard_row_ix, shard_col_ix = shard_position
        row_indices, col_indices = dataset.get_submatrix_indices(shard_row_ix, shard_col_ix, n_shards_row, n_shards_col)
        co_occurrences = dataset.get_submatrix_co_occurrences(row_indices, col_indices)

        row_indices = row_indices.to(device)
        col_indices = col_indices.to(device)
        co_occurrences = co_occurrences.to(device)

        optimizer.zero_grad()
        loss = model(row_indices, col_indices, co_occurrences, matrix_log_sum)
        # print("loss", loss.shape)
        # print(loss)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        if verbose and step % 100 == 0:
            print("Step: {}/{} \t Loss: {}".format(step, len(shard_positions), np.mean(loss_values[-10:])))

    return loss_values


def get_swivel_embeddings(model, vocab, names, add_context=True):
    # get indexes
    # TODO eventually we'll need to deal with out-of-vocab names better
    ixs = [vocab.get(name, 0) for name in names]

    emb = model.wi.weight.data[ixs].cpu().numpy()
    if add_context:
        emb += model.wj.weight.data[ixs].cpu().numpy()
    return emb


def get_best_swivel_matches(model,
                            vocab,
                            input_names,
                            candidate_names,
                            k,
                            batch_size=512,
                            add_context=True,
                            n_jobs=None,
                            progress_bar=True):
    """
    Get the best k matches for the input names from the candidate names using the glove model
    :param model: glove model
    :param vocab: map from name to id used in the model
    :param input_names: list of names for which to get matches
    :param candidate_names: list of candidate names
    :param k: number of matches to get
    :param batch_size: batch size
    :param add_context: add the context vector if true
    :param n_jobs: number of jobs to use in parallel
    :param progress_bar: display progress bar
    :return:
    """
    # Get embeddings for input names
    input_name_embeddings = get_swivel_embeddings(model, vocab, input_names, add_context=add_context)

    # Get embeddings for candidate names
    candidate_name_embeddings = get_swivel_embeddings(model, vocab, candidate_names, add_context=add_context)

    # consider euclidean???
    return get_best_matches(
        input_name_embeddings, candidate_name_embeddings, candidate_names,
        num_candidates=k, metric="cosine", batch_size=batch_size, n_jobs=n_jobs,
        progress_bar=progress_bar
    )

