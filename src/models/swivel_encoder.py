from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data import constants
from src.models.utils import convert_names_to_ids, check_convert_tensor, build_token_idx_maps

MAX_NAME_LENGTH = 30
char_to_idx_map, idx_to_char_map = build_token_idx_maps()


def convert_names_to_model_inputs(names: Union[list, np.ndarray]) -> torch.Tensor:
    """
    Return a torch tensor of names, where each name has been converted to a sequence of ids and the ids have been one-hot encoded.
    Also return the tensor where the names have been converted to a sequence of ids but before the ids have been one-hot encoded.
    :param names: list of names to encode
    :param char_to_idx_map: map characters to ids
    :param max_name_length: maximum name length
    :return: 2D tensor [names, char position]
    """
    X_targets = convert_names_to_ids(names, char_to_idx_map, MAX_NAME_LENGTH)

    return check_convert_tensor(X_targets)


class SwivelEncoderModel(nn.Module):
    def __init__(self, n_layers: int, char_embed_dim: int, n_hidden_units: int, output_dim: int, bidirectional=False, pack=False, device="cpu"):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.n_directions = (2 if bidirectional else 1)
        self.pack = pack
        self.device = device

        if char_embed_dim == 0:
            char_embed_dim = constants.VOCAB_SIZE + 1
            self.char_embedding = None
        else:
            input_size = char_embed_dim
            self.char_embedding = nn.Embedding(
                num_embeddings=constants.VOCAB_SIZE + 1,
                embedding_dim=char_embed_dim,
            )

        self.lstm = nn.LSTM(
            input_size=char_embed_dim,
            hidden_size=n_hidden_units,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.linear = nn.Linear(n_hidden_units * self.n_directions, output_dim)

        nn.init.xavier_normal_(self.linear.weight.data)

    def forward(self, X):
        """
        Generate embeddings for X
        :param X: [batch size, seq length]
        :return: [batch size, seq embedding]
        """
        X = X.to(device=self.device)
        batch_size, seq_len = X.size()

        # init hidden state before each batch
        hidden = (
            torch.randn(self.n_layers * self.n_directions, batch_size, self.n_hidden_units).to(device=self.device),  # initial hidden state
            torch.randn(self.n_layers * self.n_directions, batch_size, self.n_hidden_units).to(device=self.device),  # initial cell state
        )

        # TODO instead of packing, consider truncating to the longest length
        # TODO when packing, outputs must be re-ordered to match the original inputs; then re-consider packing
        # sort batch by sequence length
        if self.pack:
            X_lengths = torch.count_nonzero(X, dim=1).to(device="cpu").type(torch.int64)
            ixs = torch.argsort(X_lengths, descending=True)
            X = X[ixs]
            X_lengths = X_lengths[ixs]

        # get char embeddings
        if self.char_embedding is None:
            eye = torch.eye(constants.VOCAB_SIZE + 1).to(device=self.device)
            X = eye[X]
        else:
            X = self.char_embedding(X)

        # pack sequences
        if self.pack:
            X = pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=True)

        # run through LSTM
        _, (hidden, _) = self.lstm(X, hidden)

        if self.n_directions == 1:
            last_hidden = hidden[-1, :, :]
        else:
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # run through linear layer
        output = self.linear(last_hidden)  # compute the model using the hidden state of the last layer

        return output


def train_swivel_encoder(model, X_train, X_targets, num_epochs=100, batch_size=64, lr=0.01,
                         use_adam_opt=False, use_mse_loss=False, verbose=True, optimizer=None):
    """
    Train the SwivelEncoder
    :param model: SwivelEncoder model
    :param X_train: list of names
    :param X_targets: list of embeddings
    :param num_epochs: number of epochs
    :param batch_size: batch size
    :param lr: learning rate
    :param use_adam_opt: if True, use Adam optimizer; otherwise use Adagrad optimizer
    :param use_mse_loss: if True, use mean squared error (euclidean distance) loss; otherwise use cosine similarity loss
    :param verbose:print average loss every so often
    :param optimizer: passed-in optimizer to use
    """
    model = model.to(device=model.device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) \
            if use_adam_opt \
            else torch.optim.Adagrad(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss() if use_mse_loss else torch.nn.CosineEmbeddingLoss()

    dataset_train = torch.utils.data.TensorDataset(X_train, X_targets)
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    losses = list()

    for epoch in range(1, num_epochs+1):
        for batch_num, (train_batch, targets_batch) in enumerate(data_loader):
            # Clear out gradient
            model.zero_grad()

            # Compute forward pass
            x_prime = model(train_batch.to(device=model.device))

            # Compute loss do the backward pass and update parameters
            if use_mse_loss:
                loss = loss_fn(x_prime, targets_batch.to(device=model.device))
            else:
                loss = loss_fn(x_prime, targets_batch.to(device=model.device), torch.ones(len(x_prime)).to(device=model.device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Update loss value on progress bar
            if verbose:
                if batch_num % 1000 == 0:
                    print("Epoch: {}/{} \t Batch: {} \t Loss: {}".format(epoch, num_epochs, batch_num, np.mean(losses[-100:])))
    return losses
