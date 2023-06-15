# torch
import torch

# project
from src import nn as src_nn


class MLPBase(torch.nn.Module):
    def __init__(
            self,
            out_channels: int,
            hidden_channels: int,
            no_layers: int,
            linear_cls: type[torch.nn.Module],
            nonlinear_cls: type[torch.nn.Module],
            norm_cls: type[torch.nn.Module],
            bias: bool,
    ):
        super().__init__()

        # Hidden layers:
        hidden_layers = []
        for _ in range(no_layers - 2):
            hidden_layers.extend(
                [
                    linear_cls(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        bias=bias),
                    norm_cls(hidden_channels),
                    nonlinear_cls(),
                ]
            )
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.output_linear = linear_cls(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=bias)

        self.initialize(nonlinear_cls)

    def forward(self, x):
        out = self.input_layers(x)
        out = self.hidden_layers(out)
        return self.output_linear(out)

    def initialize(self, nonlinear_cls: type[torch.nn.Module]):
        # Define the gain
        if nonlinear_cls == torch.nn.ReLU:
            nonlin = "relu"
        elif nonlinear_cls == torch.nn.LeakyReLU:
            nonlin = "leaky_relu"
        else:
            nonlin = "linear"
        # Initialize hidden layers
        for (i, m) in enumerate(self.hidden_layers.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, 0, 1e-6)
        # Initialize output layer
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        if self.output_linear.bias is not None:
            torch.nn.init.normal_(self.output_linear.bias, 0, 1e-6)


class MLP(MLPBase):
    def __init__(
            self,
            data_dim: int,
            out_channels: int,
            hidden_channels: int,
            no_layers: int,
            nonlinear_cls: type[torch.nn.Module],
            norm_cls: type[torch.nn.Module],
            bias: bool,
            **kwargs,
    ):
        # Define the linear class based on the data dimension
        linear_cls = getattr(src_nn, f"Linear{data_dim}d")
        # Construct the hidden and out layers of the network
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
        input_layers = [
            linear_cls(in_channels=data_dim, out_channels=hidden_channels, bias=bias),
            norm_cls(hidden_channels),
            nonlinear_cls(),
        ]
        self.input_layers = torch.nn.Sequential(*input_layers)
        # Initialize the input layers
        self.initialize_input_layers(nonlinear_cls)

    def initialize_input_layers(self, nonlinear_cls: type[torch.nn.Module]):
        # Define the gain
        if nonlinear_cls == torch.nn.ReLU:
            nonlin = "relu"
        elif nonlinear_cls == torch.nn.LeakyReLU:
            nonlin = "leaky_relu"
        else:
            nonlin = "linear"
        # Initialize hidden layers
        for (i, m) in enumerate(self.input_layers.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, 0, 1e-6)
