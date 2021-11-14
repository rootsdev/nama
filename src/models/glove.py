from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models.utils import get_best_matches


class GloveDataset:

    def __init__(self, input_names, weighted_actual_names, vocab_size=200000, device="cpu"):
        # get the most-frequent names from weighted_actual_names
        name_counter = Counter()
        for input_name, wan in zip(input_names, weighted_actual_names):
            for name, _, co_occurrence in wan:
                name_counter[input_name] += co_occurrence
                name_counter[name] += co_occurrence
        self._word2id = {w: i for i, (w, _) in enumerate(name_counter.most_common(vocab_size))}
        self._id2word = {i: w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)

        # create co-occurrence matrix
        cooc_mat = defaultdict(lambda: defaultdict(int))
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
                cooc_mat[input_id][name_id] += co_occurrence
                # symmetric
                cooc_mat[name_id][input_id] += co_occurrence

        # create tensors
        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        for name1, matches in cooc_mat.items():
            for name2, cnt in matches.items():
                self._i_idx.append(name1)
                self._j_idx.append(name2)
                self._xij.append(cnt)

        self._i_idx = torch.LongTensor(self._i_idx).to(device=device)
        self._j_idx = torch.LongTensor(self._j_idx).to(device=device)
        self._xij = torch.FloatTensor(self._xij).to(device=device)

        print("Vocabulary length: {}".format(self._vocab_len))

    def get_vocab(self):
        return self._word2id

    def get_batches(self, batch_size):
        # Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p + batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x


def _weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx


def _wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss)


def train_glove(model, dataset, n_epochs=100, batch_size=2048, x_max=100, alpha=0.75, lr=0.05, device="cpu"):
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    n_batches = int(len(dataset._xij) / batch_size)
    loss_values = list()

    for e in range(1, n_epochs+1):
        batch_num = 0

        for x_ij, i_idx, j_idx in dataset.get_batches(batch_size):
            batch_num += 1

            optimizer.zero_grad()
            outputs = model(i_idx, j_idx)
            weights_x = _weight_func(x_ij, x_max, alpha).to(device=device)
            loss = _wmse_loss(weights_x, outputs, torch.log(x_ij)).to(device=device)

            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

            if batch_num % 100 == 0:
                print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, n_epochs, batch_num, n_batches, np.mean(loss_values[-20:])))

    return loss_values


def get_glove_embeddings(model, vocab, names, p=1):
    # normalize embeddings column-wise
    norm_wi = F.normalize(model.wi.weight.data, p=p, dim=0).cpu().numpy()
    norm_wj = F.normalize(model.wj.weight.data, p=p, dim=0).cpu().numpy()

    # get indexes - eventually we'll need to deal with out-of-vocab names better
    ixs = [vocab.get(name, 0) for name in names]

    emb_i = norm_wi[ixs]
    emb_j = norm_wj[ixs]
    return emb_i + emb_j


def get_best_glove_matches(model, vocab, input_names, candidate_names, k, batch_size=512, n_jobs=None):
    """
    Get the best k matches for the input names from the candidate names using the glove model
    :param model: glove model
    :param vocab: map from name to id used in the model
    :param input_names: list of names for which to get matches
    :param candidate_names: list of candidate names
    :param k: number of matches to get
    :param batch_size: batch size
    :param n_jobs: number of jobs to use in parallel
    :return:
    """
    # Get embeddings for input names
    input_name_embeddings = get_glove_embeddings(model, vocab, input_names)

    # Get embeddings for candidate names
    candidate_name_embeddings = get_glove_embeddings(model, vocab, candidate_names)

    return get_best_matches(
        input_name_embeddings, candidate_name_embeddings, candidate_names,
        num_candidates=k, metric="cosine", batch_size=batch_size, n_jobs=n_jobs
    )

