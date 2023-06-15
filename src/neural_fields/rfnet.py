# torch
import torch

import numpy as np

# project
from .mlp import MLPBase
from src import nn as src_nn


class RFNet(MLPBase):
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            hidden_channels: int,
            no_layers: int,
            nonlinear_cls: type[torch.nn.Module],
            norm_cls: type[torch.nn.Module],
            omega_0: float,
            bias: bool,
            **kwargs,
    ):
        # Define the linear class based on the data dimension
        linear_cls = getattr(src_nn, f"Linear{data_dim}d")
        # construct the hidden and out layers of the network
        super().__init__(
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            no_layers=no_layers,
            linear_cls=linear_cls,
            nonlinear_cls=nonlinear_cls,
            norm_cls=norm_cls,
            bias=bias,
        )
        # Construct input embedding
        self.input_layers = RandomFourierEmbedding(
            data_dim=data_dim,
            out_channels=hidden_channels,
            omega_0=omega_0,
            bias=bias,
        )


class RandomFourierEmbedding(torch.nn.Module):
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            omega_0: float,
            bias: int,
    ):
        super().__init__()

        assert (
                out_channels % 2 == 0
        ), f"out_channels must be even. Current {out_channels}"
        linear_out_channels = out_channels // 2
        # Define the linear class based on the data dimension
        linear_cls = getattr(src_nn, f"Linear{data_dim}d")
        self.linear = linear_cls(in_channels=data_dim, out_channels=linear_out_channels, bias=bias)
        # Initialize:
        torch.nn.init.normal_(self.linear.weight, 0.0, 2 * np.pi * omega_0)
        self.omega_0 = omega_0

    def forward(self, x):
        out = self.linear(x)
        return torch.cat([torch.cos(out), torch.sin(out)], dim=1)
