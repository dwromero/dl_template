# torch
import torch

from math import sqrt

# project
from src import nn as src_nn


class SIRENBase(torch.nn.Module):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        linear_cls: type[torch.nn.Module],
        omega_0: float,
        bias: bool,
        learn_omega_0: bool,
    ):
        super().__init__()

        # 1st layer:
        kernel_net = [
            SirenLayer(
                in_channels=data_dim,
                out_channels=hidden_channels,
                linear_cls=linear_cls,
                omega_0=omega_0,
                bias=bias,
                learn_omega_0=learn_omega_0,
            )
        ]
        # Hidden layers:
        for _ in range(no_layers - 2):
            kernel_net.extend(
                [
                    SirenLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        linear_cls=linear_cls,
                        omega_0=omega_0,
                        learn_omega_0=learn_omega_0,
                        bias=bias,
                    )
                ]
            )
        self.kernel_net = torch.nn.Sequential(*kernel_net)
        # Last layer:
        self.output_linear = linear_cls(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=bias)
        # initialize the kernel function
        self.initialize(omega_0=omega_0)

    def forward(self, x):
        out = self.kernel_net(x)
        return self.output_linear(out)

    def initialize(self, omega_0):
        net_layer = 1
        for (i, m) in enumerate(self.kernel_net.modules()):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                if net_layer == 1:
                    w_std = 1 / m.weight.shape[1]
                    torch.nn.init.uniform_(m.weight, -w_std, w_std) # TODO: Check!
                    net_layer += 1
                else:
                    w_std = sqrt(6.0 / m.weight.shape[1]) / omega_0
                    torch.nn.init.uniform_(m.weight, -w_std, w_std)
                if m.bias is not None:
                    # Important! Bias is not defined in original SIREN implementation!
                    torch.nn.init.normal_(m.bias, 0, 1e-6)
        # The final layer must be initialized differently because it is not multiplied by omega_0
        torch.nn.init.kaiming_uniform_(self.output_linear.weight, nonlinearity="linear")
        if self.output_linear.bias is not None:
            torch.nn.init.normal_(self.output_linear.bias, 0, 1e-6)


#############################################
#       SIREN as in Sitzmann et al., 2020
##############################################
class SIREN(SIRENBase):
    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_channels: int,
        no_layers: int,
        omega_0: float,
        bias: bool,
        learn_omega_0: bool,
        **kwargs,
    ):
        linear_cls = getattr(src_nn, f"Linear{data_dim}d")
        super().__init__(
            data_dim=data_dim,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            no_layers=no_layers,
            linear_cls=linear_cls,
            omega_0=omega_0,
            bias=bias,
            learn_omega_0=learn_omega_0)


class SirenLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        linear_cls: type[torch.nn.Module],
        omega_0: float,
        learn_omega_0: bool,
        bias: bool,
    ):
        """Implements a Linear Layer of the form y = omega_0 * [W x + b] where x is 1 dimensional."""
        super().__init__()
        self.linear = linear_cls(in_channels=in_channels, out_channels=out_channels, bias=bias)
        # omega_0
        if learn_omega_0:
            self.omega_0 = torch.nn.Parameter(torch.Tensor(1))
            with torch.no_grad():
                self.omega_0.fill_(omega_0)
        else:
            tensor_omega_0 = torch.zeros(1)
            tensor_omega_0.fill_(omega_0)
            self.register_buffer("omega_0", tensor_omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

    def extra_repr(self):
        return (
            super().extra_repr() + f" omega_0={self.omega_0.item():.2f}, "
            f"learn_omega_0={self.omega_0.requires_grad}"
        )
