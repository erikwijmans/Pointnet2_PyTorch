import hydra
import pytorch_lightning as pl
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


@hydra.main("config/config.yaml")
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, cfg)

    early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        filepath=cfg.task_model.name,
        verbose=True,
    )
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
