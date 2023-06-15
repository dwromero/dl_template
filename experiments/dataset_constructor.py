import os

# torch
import torch
import pytorch_lightning as pl
from ml_collections import config_dict

# project
from src import datamodules


def construct_datamodule(cfg: config_dict.ConfigDict) -> pl.LightningDataModule:
    # Define num_workers
    if cfg.num_workers == -1:
        cfg.num_workers = int(os.cpu_count() / torch.cuda.device_count())

    # Define pin_memory
    pin_memory = True if cfg.device == "cuda" and torch.cuda.is_available() else False

    # Gather module from datamodules & create instance
    dataset_name = f"{cfg.dataset.name}DataModule"
    dataset = getattr(datamodules, dataset_name)
    datamodule = dataset(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size // cfg.train.accumulate_grad_steps,
        test_batch_size=cfg.test.batch_size_multiplier * cfg.train.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        augment=cfg.dataset.augment,
        **cfg.dataset.params,
    )

    # All datamodules must define these variables for model creation
    assert hasattr(datamodule, "data_dim")
    assert hasattr(datamodule, "input_channels")
    assert hasattr(datamodule, "output_channels")
    assert hasattr(datamodule, "data_type")
    return datamodule
