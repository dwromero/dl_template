# torch
import torch

import numpy as np

# project
from src import nn as src_nn


class MFNBase(torch.nn.Module):
    """Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            hidden_channels: int,
            no_layers: int,
            bias: bool,
    ):
        super().__init__()
        # Define the linear class based on the data dimension
        linear_cls = getattr(src_nn, f"Linear{data_dim}d")
        # Hidden layers
        self.linears = torch.nn.ModuleList(
            [
                linear_cls(in_channels=hidden_channels,
                           out_channels=hidden_channels,
                           bias=bias)
                for _ in range(no_layers)
            ]
        )
        # Final layer
        self.output_linear = linear_cls(in_channels=hidden_channels,
                                        out_channels=out_channels,
                                        bias=bias)
        self.initialize()

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linears[i - 1](out)
        out = self.output_linear(out)
        return out

    def initialize(self):
        for linear in self.linears:
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity="linear")
            if linear.bias is not None:
                torch.nn.init.normal_(linear.bias, 0.0, 1e-6)
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        if self.output_linear.bias is not None:
            torch.nn.init.normal_(self.output_linear.bias, 0.0, 1e-6)


#############################################
#       FourierNet
##############################################
class FourierLayer(torch.nn.Module):
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            omega_0: float,
            bias: bool,
    ):
        super().__init__()
        # Define the linear class based on the data dimension
        linear_cls = getattr(src_nn, f"Linear{data_dim}d")
        self.linear = linear_cls(data_dim, out_channels, bias)
        self.omega_0 = omega_0
        self.initialize()

    def initialize(self):
        # Initialize
        w_std = 1.0 / self.linear.weight.shape[1]
        w_std *= 2.0 * np.pi * self.omega_0
        torch.nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if self.linear.bias is not None:
            torch.nn.init.normal_(self.linear.bias, 0.0, 1e-6)

    def forward(self, x):
        return torch.sin(self.linear(x))

    def extra_repr(self):
        return f"omega_0={self.omega_0}"


class FourierNet(MFNBase):
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            hidden_channels: int,
            no_layers: int,
            omega_0: float,
            bias: bool,
            **kwargs,
    ):
        super().__init__(
            data_dim=data_dim,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            no_layers=no_layers,
            bias=bias,
        )
        self.filters = torch.nn.ModuleList(
            [
                FourierLayer(data_dim=data_dim,
                             out_channels=hidden_channels,
                             omega_0=omega_0,
                             bias=bias)
                for _ in range(no_layers + 1)
            ]
        )


#############################################
#       GaborNet
##############################################
def gaussian_window(x, gamma, mu):
    return torch.exp(-0.5 * ((gamma * (x.unsqueeze(1) - mu)) ** 2).sum(2))


class GaborLayer(torch.nn.Module):
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            omega_0: float,
            alpha: float,
            beta: float,
            bias: bool,
    ):
        super().__init__()
        # Define the linear class based on the data dimension
        linear_cls = getattr(src_nn, f"Linear{data_dim}d")
        # Construct & initialize parameters
        mu = 2 * torch.rand(out_channels, data_dim) - 1
        gamma = torch.distributions.gamma.Gamma(alpha, beta).sample((out_channels, 1))  # Isotropic
        self.mu = torch.nn.Parameter(mu)
        self.gamma = torch.nn.Parameter(gamma)
        # Create and initialize parameters
        self.linear = linear_cls(data_dim, out_channels, bias=bias)
        self.data_dim = data_dim
        self.omega_0 = omega_0
        self.alpha = alpha
        self.beta = beta
        self.initialize()

    def initialize(self):
        w_std = 1.0 / self.linear.weight.shape[1]
        self.linear.weight.data.uniform_(-w_std, w_std)
        self.linear.weight.data *= 2.0 * torch.pi * self.omega_0 * \
                                   self.gamma.view(*self.gamma.shape, *((1,) * self.data_dim))
        if self.linear.bias is not None:
            torch.nn.init.normal_(self.linear.bias, 0.0, 1e-6)

    def forward(self, x):
        gauss_window = gaussian_window(
            x,
            self.gamma.view(
                1, *self.gamma.shape, *((1,) * self.data_dim)
            ),  # TODO. We can avoid doing this
            self.mu.view(1, *self.mu.shape, *((1,) * self.data_dim)),
        )
        return gauss_window * torch.sin(self.linear(x))


class GaborNet(MFNBase):
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            hidden_channels: int,
            no_layers: int,
            omega_0: float,
            bias: bool,
            alpha: float = 6.0,
            beta: float = 1.0,
            **kwargs,
    ):
        super().__init__(
            data_dim=data_dim,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            no_layers=no_layers,
            bias=bias,
        )
        self.filters = torch.nn.ModuleList(
            [
                GaborLayer(data_dim=data_dim,
                           out_channels=hidden_channels,
                           omega_0=omega_0,
                           alpha=alpha / (no_layers + 1),
                           beta=beta,
                           bias=bias)
                for _ in range(no_layers + 1)
            ]
        )
