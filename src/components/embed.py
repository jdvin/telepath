import torch
import torch.nn as nn


class Embedding3D(nn.Module):
    def __init__(self, num_embeddings, embedding_dim1, embedding_dim2):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim1, embedding_dim2)
        )
        nn.init.normal_(self.weight, mean=0, std=0.02)

    def forward(self, input):
        # Input shape: (batch_size, seq_len)
        # Output shape: (batch_size, seq_len, embedding_dim1, embedding_dim2)
        return torch.index_select(self.weight, 0, input.view(-1)).view(
            *input.shape, self.weight.size(1), self.weight.size(2)
        )
