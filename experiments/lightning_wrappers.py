import glob
import wandb

# torch
import torch
import torchmetrics
import pytorch_lightning as pl
from ml_collections import config_dict

# project
import experiments
import src
import src.nn as src_nn


class LightningWrapperBase(pl.LightningModule):
    def __init__(
            self,
            network: torch.nn.Module,
            cfg: config_dict.ConfigDict,
    ):
        super().__init__()
        # Define network
        self.network = network
        # Save optimizer & scheduler parameters
        self.optim_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        # Regularization metrics
        if self.optim_cfg.weight_decay != 0.0:
            self.weight_regularizer = src.LnLoss(
                weight_loss=self.optim_cfg.weight_decay, norm_type=2
            )
        else:
            self.weight_regularizer = None
        # Placeholders for logging of best train & validation values
        self.num_params = -1
        # Explicitly define whether we are in distributed mode.
        self.distributed = cfg.train.distributed and cfg.train.avail_gpus != 1

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        # Construct optimizer & scheduler
        optimizer = experiments.construct_optimizer(
            model=self, optim_cfg=self.optim_cfg
        )
        scheduler = experiments.construct_scheduler(
            optimizer=optimizer, scheduler_cfg=self.scheduler_cfg
        )
        # Construct output dictionary
        output_dict = {"optimizer": optimizer}
        if scheduler is not None:
            output_dict["lr_scheduler"] = {}
            output_dict["lr_scheduler"]["scheduler"] = scheduler
            output_dict["lr_scheduler"]["interval"] = "step"

            # If we use a ReduceLROnPlateu scheduler, we must monitor val/acc
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.scheduler_cfg.mode == "min":
                    output_dict["lr_scheduler"]["monitor"] = "val/loss"
                else:
                    output_dict["lr_scheduler"]["monitor"] = "val/acc"
                output_dict["lr_scheduler"]["reduce_on_plateau"] = True
                output_dict["lr_scheduler"]["interval"] = "epoch"
            # TODO(dwromero): ReduceLROnPlateau with warmup
            if isinstance(
                    scheduler, src_nn.schedulers.ChainedScheduler
            ) and isinstance(
                scheduler._schedulers[-1], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                raise NotImplementedError("cannot use ReduceLROnPlateau with warmup")
        return output_dict

    def on_train_start(self):
        if type(self.logger) == pl.loggers.logger.DummyLogger:
            return  # running in fast_dev_run mode, skip as logger not usable
        if self.global_rank == 0:
            # Calculate and log the size of the model
            if self.num_params == -1:
                with torch.no_grad():
                    # Log parameters
                    no_params = src.utils.num_params(self.network)
                    self.logger.experiment.summary["no_params"] = no_params
                    self.num_params = no_params
                    # Log source code files
                    code = wandb.Artifact(
                        f"source-code-{self.logger.experiment.name}", type="code"
                    )
                    # Get paths of all source code files
                    paths = glob.glob("**/*.py", recursive=True)
                    paths += glob.glob("**/*.yaml", recursive=True)
                    # Filter paths
                    paths = list(filter(lambda x: "wandb" not in x, paths))
                    # Get all source files
                    for path in paths:
                        code.add_file(path, name=path)
                    # Use the artifact
                    if not self.logger.experiment.offline:
                        wandb.run.use_artifact(code)


class ClassificationWrapper(LightningWrapperBase):
    def __init__(
            self,
            network: torch.nn.Module,
            cfg: config_dict.ConfigDict,
            **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)

        # Define metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=cfg.net.out_channels)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=cfg.net.out_channels)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                              num_classes=cfg.net.out_channels)
        self.loss_metric = torch.nn.CrossEntropyLoss()

        # Caches for step responses
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Placeholders for logging of best train & validation values
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0

        # Compute predictions
        self.get_predictions = lambda logits: torch.argmax(logits, 1)

    def _step(self, batch, accuracy_calculator):
        x, labels = batch
        logits = self(x)
        # Predictions
        predictions = self.get_predictions(logits)
        # Calculate accuracy and loss
        accuracy_calculator(predictions, labels)
        loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return predictions, logits, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.train_acc)
        # Add regularization
        if self.weight_regularizer is not None:
            reg_loss = self.weight_regularizer(self.network)
        else:
            reg_loss = 0.0
        # Log and return loss (Required in training step)
        self.log(
            "train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "train/reg_loss",
            reg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.training_step_outputs.append(logits.detach())
        return loss + reg_loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.val_acc)
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.validation_step_outputs.append(logits)
        return logits  # used to log histograms in validation_epoch_step

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, _, loss = self._step(batch, self.test_acc)
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )

    def on_training_epoch_end(self):
        flattened_logits = torch.cat(self.training_step_outputs)
        flattened_logits = torch.flatten(flattened_logits)
        self.logger.experiment.log(
            {
                "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )
        self.training_step_outputs.clear()
        # Log best accuracy
        train_acc = self.trainer.callback_metrics["train/acc_epoch"]
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc.item()
            self.logger.experiment.log(
                {
                    "train/best_acc": self.best_train_acc,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Gather logits from validation set and construct a histogram of them.
        flattened_logits = torch.flatten(torch.cat(self.validation_step_outputs))
        self.logger.experiment.log(
            {
                "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                "global_step": self.global_step,
            }
        )
        self.validation_step_outputs.clear()
        # Log best accuracy
        val_acc = self.trainer.callback_metrics["val/acc"]
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            self.logger.experiment.log(
                {
                    "val/best_acc": self.best_val_acc,
                    "global_step": self.global_step,
                }
            )


class RegressionWrapper(LightningWrapperBase):
    def __init__(
            self,
            network: torch.nn.Module,
            cfg: config_dict.ConfigDict,
            **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)

        metric = cfg.train.metric

        # Define metrics
        if metric == "MAE":
            metric_cls = torchmetrics.MeanAbsoluteError
            loss_metric_cls = torch.nn.L1Loss
        elif metric == "MSE":
            metric_cls = torchmetrics.MeanSquaredError
            loss_metric_cls = torch.nn.MSELoss
        else:
            raise ValueError(f"Metric {metric} not recognized")
        self.train_metric = metric_cls()
        self.val_metric = metric_cls()
        self.test_metric = metric_cls()
        self.loss_metric = loss_metric_cls()

        # Caches for step responses
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Placeholders for logging of best train & validation values
        self.best_train_loss = 1e9
        self.best_val_loss = 1e9

    def _step(self, batch, metric_calculator):
        x, groud_truth = batch
        prediction = self(x)
        # Calculate loss
        metric_calculator(prediction, groud_truth)
        loss = self.loss_metric(prediction, groud_truth)
        # Return predictions and loss
        return prediction, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._step(batch, self.train_metric)
        # Add regularization
        if self.weight_regularizer is not None:
            reg_loss = self.weight_regularizer(self.network)
        else:
            reg_loss = 0.0
        # Log and return loss (Required in training step)
        self.log(
            "train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed
        )
        self.log(
            "train/reg_loss",
            reg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        return loss + reg_loss

    def validation_step(self, batch, batch_idx):
        predictions, loss = self._step(batch, self.val_metric)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )

    def test_step(self, batch, batch_idx):
        predictions, loss = self._step(batch, self.test_metric)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )

    def on_training_epoch_end(self):
        # Log best accuracy
        train_loss = self.trainer.callback_metrics["train/loss_epoch"]
        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            self.logger.experiment.log(
                {
                    "train/best_loss": self.best_train_loss,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Log best accuracy
        val_loss = self.trainer.callback_metrics["val/loss"]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.logger.experiment.log(
                {
                    "val/best_loss": self.best_val_loss,
                    "global_step": self.global_step,
                }
            )
