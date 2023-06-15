# torch
import torch
import torchvision


class ResNet18(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = torch.nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet18.fc = torch.nn.Linear(
            in_features=self.resnet18.fc.in_features, out_features=out_channels
        )

    def forward(self, x):
        return self.resnet18(x)


if __name__ == "__main__":
    model = ResNet18(in_features=1, num_classes=10)
    print(model)