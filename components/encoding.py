import math

xw
import torch
import numpy as np


class PositionalEncoder(torch.nn.Module):
    def get_angles(pos: torch.Tensor, k: torch.Tensor, d: int) -> torch.Tensor:
        """
        Get the angles for the positional encoding

        Argss:
            pos: Column vector containing the positions [[0], [1], ...,[N-1]]
            k: Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
            d(integer): Encoding size
        """

        # Get i from dimension span k
        i = k // 2
        # Calculate the angles using pos, i and d
        angles = pos / (10000 ** ((2 * i) / d))

        return angles

    def __init__(self, d_model, max_seq_len):
        """
        Precomputes a matrix with all the positional encodings

        Arguments:
            d_model (int) -- Encoding size
            max_seq_len (int) -- Maximum number of positions to be encoded
        """
        super().__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        pe = self.get_angles(
            np.arange(max_seq_len)[:, np.newaxis], np.arange(d_model), d_model
        )

        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = np.sin(pe[:, 0::2])

        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add constant to embedding
        seq_len = x.size(1)
        x = x + torch.Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x
