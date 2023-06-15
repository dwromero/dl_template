### Deep Learning Template

Here's an example:
```
--cfg=cfg/default_config.py --cfg.num_workers=1 --cfg.dataset.name=CIFAR10 --cfg.dataset.augment=True --cfg.train.epochs=200 --cfg.train.batch_size=50 --cfg.net.type=ResNet18 --cfg.optimizer.type=Adam --cfg.optimizer.lr=1e-3 --cfg.scheduler.type=cosine --cfg.scheduler.warmup_epochs=10
```

