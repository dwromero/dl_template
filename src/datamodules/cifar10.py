# torch
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        num_workers,
        pin_memory,
        **kwargs,
    ):
        super().__init__()

        # Save parameters to self
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.grayscale = kwargs["grayscale"]
        self.augment = kwargs["augment"]

        self.data_type = "image"
        self.data_dim = 2

        # Determine sizes of dataset
        if self.grayscale:
            self.input_channels = 1
        else:
            self.input_channels = 3
        self.output_channels = 10

        # Create transforms
        if self.grayscale:
            train_transform = [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            ]
        else:
            train_transform = [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.247, 0.243, 0.261),
                ),
            ]

        val_test_transform = train_transform
        # Augmentation before normalization, taken from:
        # https://github.com/dipuk0506/SpinalNet/blob/master/CIFAR-10/ResNet_default_and_SpinalFC_CIFAR10.py#L39
        if self.augment:
            train_transform = [
                transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
                transforms.RandomHorizontalFlip(),
            ] + train_transform

        self.train_transform = transforms.Compose(train_transform)
        self.val_test_transform = transforms.Compose(val_test_transform)

    def prepare_data(self):
        # download data, train then test
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            cifar10 = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.train_transform
            )
            self.train_dataset, self.val_dataset = random_split(
                cifar10,
                [45000, 5000],
                generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
            )
            self.train_dataset.dataset.transform = self.train_transform
            self.val_dataset.dataset.transform = self.val_test_transform  # no augmentation
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.val_test_transform
            )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return test_dataloader
