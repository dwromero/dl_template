from ml_collections import config_dict

# torch
import torch
import pytorch_lightning as pl

# project
import src
from experiments import lightning_wrappers


def construct_model(
        cfg: config_dict.ConfigDict, datamodule: pl.LightningDataModule
) -> pl.LightningModule:
    # Get parameters of model from task type
    data_dim = datamodule.data_dim
    in_channels = datamodule.input_channels
    out_channels = datamodule.output_channels

    # Get type of model from task type
    net_type = f"{cfg.net.type}"

    # Overwrite data_dim in cfg.net
    cfg.net.data_dim = data_dim
    cfg.net.in_channels = in_channels
    cfg.net.out_channels = out_channels

    # Print automatically derived model parameters.
    print(
        f"Automatic Parameters:\n dataset = {cfg.dataset.name}, "
        f"net_name = {net_type},"
        f" data_dim = {data_dim}"
        f" in_channels = {in_channels},"
        f" out_chanels = {out_channels}."
    )

    # Create and return model
    net_type = getattr(src.models, net_type)
    network = net_type(in_channels=in_channels,
                       out_channels=out_channels,
                       net_cfg=cfg.net,
                       conv_cfg=cfg.conv)

    # Wrap the network in a LightningModule.
    model = lightning_wrappers.ClassificationWrapper(network=network, cfg=cfg)
    return model
