from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    with_statement,
)

import argparse
import os
import os.path as osp
import pprint

import hydra
import pytorch_lightning as pl
import torch

from pointnet2.models import models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


@hydra.main("config/config.yaml")
def main(cfg):
    model = models[f"{cfg.model}-{cfg.task}"](cfg)

    early_stop_callback = pl.callbacks.EarlyStopping(patience=20)
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_nb_epochs=cfg.epochs,
        early_stop_callback=early_stop_callback,
        distributed_backend=cfg.distrib_backend,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
