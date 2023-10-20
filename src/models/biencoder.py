import torch
from torch import nn
import torch.nn.functional as F


class BiEncoder(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_tokens, pad_token, pretrained_embeddings=None):
        super(BiEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self.pad_token = pad_token
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.positional_embedding = nn.Embedding(num_embeddings=max_tokens, embedding_dim=embedding_dim)
        # self.forward_positional_embedding = nn.Embedding(num_embeddings=max_tokens+1, embedding_dim=embedding_dim)
        # self.backward_positional_embedding = nn.Embedding(num_embeddings=max_tokens+1, embedding_dim=embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pooling layer to create a single vector

    def forward(self, input):
        # get token embedding
        embedded = self.embedding(input)  # Shape: (batch_size, max_tokens, embedding_dim)
        # get mask
        mask = torch.where(input == self.pad_token, 0, 1)[..., None]  # Shape: (batch_size, max_tokens, 1)
        # get positional embedding
        positions = torch.arange(start=0, end=self.max_tokens).repeat(input.shape[0], 1)
        positional_embedded = self.positional_embedding(positions)
        #         # get forward positional embedding: pad token is position 0
        #         positions = torch.arange(start=1, end=self.max_tokens+1).repeat(input.shape[0], 1)
        #         forward_positions = torch.where(input == self.pad_token, 0, positions)
        #         forward_positional_embedded = self.forward_positional_embedding(forward_positions)
        # get backward positional embedding
        #         backward_positions = torch.where(input == self.pad_token, 0, 1)
        #         backward_n_tokens = backward_positions.sum(dim=1)
        #         for ix in range(backward_n_tokens.shape[0]):
        #             n_tokens = backward_n_tokens[ix]
        #             backward = torch.arange(start=n_tokens, end=0, step=-1)
        #             backward_positions[ix][:n_tokens] = backward
        #         backward_positional_embedded = self.backward_positional_embedding(backward_positions)
        # multiply embeddings
        #         embedded = embedded * forward_positional_embedded * backward_positional_embedded
        embedded = embedded * positional_embedded * mask
        pooled = self.pooling(embedded.permute(0, 2, 1)).squeeze(2)  # Shape: (batch_size, embedding_dim)
        return pooled

    def get_embedding(self, name_tokens):
        embedding = self(torch.tensor([name_tokens]))[0]
        return embedding.detach().numpy()

    def predict(self, name1_tokens, name2_tokens):
        self.eval()
        with torch.no_grad():
            embeddings = self(torch.tensor([name1_tokens, name2_tokens]))
        # return (embeddings[0] * embeddings[1]).sum(dim=-1).item()
        return F.cosine_similarity(embeddings[0], embeddings[1], dim=-1).item()
