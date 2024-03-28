import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, max_step=1000):
        super().__init__()
        # Create a long enough P
        self.P = torch.zeros((1, max_step, num_hiddens))
        X = torch.arange(max_step, dtype=torch.float32).reshape(-1, 1)
        X = X / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, t):
        return self.P[:, t, :].to(t.device)
