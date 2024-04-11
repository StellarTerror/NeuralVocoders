import trainmodule
import datamodule
import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
import json
import torch
import argparse


def train(config):
    train_module = trainmodule.Vocoder(config)
    data_module = datamodule.VocoderDataModule(config)

    callbacks = [plc.ModelCheckpoint(), plc.RichProgressBar()]

    trainer = pl.Trainer(max_epochs=config["learning_option"]["num_epochs"], callbacks=callbacks, precision="32")

    trainer.fit(train_module, data_module)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="config/HiFi-GAN V1.json")
    
    args = argparser.parse_args()
    config = json.load(open(args.config, "r"))

    train(config)
