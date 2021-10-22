import torch
import torch.nn as nn
from tqdm import trange

# NumPy has 64 by default, setting 64 on torch as well to avoid conflicts
torch.set_default_dtype(torch.float64)


class AutoEncoder(nn.Module):
    """
    AutoEncoder using bidirectional LSTM for the encoder, bidirectional LSTM for the decoder, followed by a single linear layer
    """

    def __init__(self, input_size, hidden_size, num_layers, seq_len, device):
        super().__init__()
        self.seq_len = seq_len
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
