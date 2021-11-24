from typing import Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from src.data import constants
from src.models.utils import convert_names_to_ids, one_hot_encode, get_best_matches, check_convert_tensor, \
    ArrayDataLoader, build_token_idx_maps

# NumPy has 64 by default, setting 64 on torch as well to avoid conflicts
torch.set_default_dtype(torch.float64)

MAX_NAME_LENGTH = 30
char_to_idx_map, idx_to_char_map = build_token_idx_maps()


def convert_names_to_model_inputs(names: Union[list, np.ndarray]) -> (torch.Tensor, torch.Tensor):
    """
    Return a torch tensor of names, where each name has been converted to a sequence of ids and the ids have been one-hot encoded.
    Also return the tensor where the names have been converted to a sequence of ids but before the ids have been one-hot encoded.
    :param names: list of names to encode
    :param char_to_idx_map: map characters to ids
    :param max_name_length: maximum name length
    :return: 3D tensor, where axis 0 is names, axis 1 is character position and axis 2 is the one-hot encoding (a..z).
             also a 2D tensor of names->ids
    """
    X_targets = convert_names_to_ids(names, char_to_idx_map, MAX_NAME_LENGTH)
    X_one_hot = one_hot_encode(X_targets, constants.VOCAB_SIZE + 1)

    X_inputs = check_convert_tensor(X_one_hot)
    X_targets = check_convert_tensor(X_targets)

    return X_inputs, X_targets


def get_embeddings(model, names: Union[list, np.ndarray], batch_size: int = 1024) -> np.ndarray:
    X, _ = convert_names_to_model_inputs(names)
    return get_embeddings_from_X(model, X, batch_size)


def get_embeddings_from_X(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    results = []
    with torch.inference_mode():
        for batch in ArrayDataLoader(X, batch_size):
            batch = check_convert_tensor(batch)
            results.append(model(batch, just_encoder=True).detach().cpu().numpy())
    return np.vstack(results)


def get_best_autoencoder_matches(model, input_names, candidate_names, k, batch_size=512, n_jobs=None):
    """
    Get the best k matches for the input names from the candidate names using the autoencoder model
    :param model: autoencoder model
    :param input_names: list of names for which to get matches
    :param candidate_names: list of candidate names
    :param k: number of matches to get
    :param batch_size: batch size
    :param n_jobs: number of jobs to use in parallel
    :return:
    """
    # Get embeddings for input names
    input_name_embeddings = get_embeddings(model, input_names, batch_size)

    # Get embeddings for candidate names
    candidate_name_embeddings = get_embeddings(model, candidate_names, batch_size)

    return get_best_matches(
        input_name_embeddings, candidate_name_embeddings, candidate_names,
        num_candidates=k, metric="euclidean", batch_size=batch_size, n_jobs=n_jobs
    )


class AutoEncoder(nn.Module):
    """
    AutoEncoder using bidirectional LSTM for the encoder, bidirectional LSTM for the decoder, followed by a single linear layer
    """

    def __init__(self, input_size, hidden_size, num_layers, device):
        super().__init__()
        self.seq_len = MAX_NAME_LENGTH
        self.device = device
        # input size is vocab size
        self.lstm_encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True
        ).to(device)
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size * 2, hidden_size=hidden_size, batch_first=True  # times 2 because bi-directional
        ).to(device)
        self.linear = nn.Linear(hidden_size, input_size).to(device)

    def forward(self, x, just_encoder=False):
        # Encode input - x_encoded is the last hidden state
        _, (x_encoded, _) = self.lstm_encoder(x.to(self.device))

        # Concatenate left-right hidden vectors (bi-directional)
        x_encoded = torch.cat([x_encoded[0], x_encoded[1]], dim=1)

        # After training is done we only need the encoded vectors
        if just_encoder:
            return x_encoded

        # Reshape data to have seq_len time steps
        # TODO why do we do this?
        x_encoded = x_encoded.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decode the encoded input - x_decoded is the output
        x_decoded, (_, _) = self.lstm_decoder(x_encoded)

        return self.linear(x_decoded)


def train_model(model, X_train, X_targets, num_epochs=100, batch_size=128):
    """
    Train the AutoEncoder
    :param model: AutoEncoder model
    :param X_train: 3D array from convert_names_to_model_input
    :param X_targets: list of names corresponding to axis 0 of X_train
    :param num_epochs: number of epochs
    :param batch_size: batch size
    """
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    dataset_train = torch.utils.data.TensorDataset(X_train, X_targets)
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    with trange(num_epochs) as pbar:
        for _ in pbar:
            for i, (train_batch, labels_batch) in enumerate(data_loader):
                # Clear out gradient
                model.zero_grad()

                # Compute forward pass
                # Reshape output to match CrossEntropyLoss input
                x_prime = model(train_batch).transpose(1, -1)

                # Compute loss do the backward pass and update parameters
                loss = loss_fn(x_prime, labels_batch.to(model.device))
                loss.backward()
                optimizer.step()

            # Update loss value on progress bar
            pbar.set_postfix(loss=loss.item())
