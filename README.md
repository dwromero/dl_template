### Deep Learning Template

## 1. Installation

## 2. Contribute

Before contributing, please make sure that `pre-commit` is installed. To this end, run the following command:

```
pip install pre-commit  # in case it is not installed
pre-commit install
```

## 3. Usage

To see the pipeline, the best is to run an example that goes through the whole code. To this end, run the following command:

```
--cfg=cfg/default_config.py --cfg.num_workers=1 --cfg.dataset.name=CIFAR10 --cfg.dataset.augment=True --cfg.train.epochs=200 --cfg.train.batch_size=50 --cfg.net.type=ResNet18 --cfg.optimizer.type=Adam --cfg.optimizer.lr=1e-3 --cfg.scheduler.type=cosine --cfg.scheduler.warmup_epochs=10
```
