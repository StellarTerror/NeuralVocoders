import trainmodule
import datamodule
import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
import json
import torch

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42)
    pl.cli_lightning_logo()

    settings = json.load(open("config/HiFi-GAN V1.json"))
    #settings = json.load(open("config/HiFi-GAN V2.json"))
    #settings = json.load(open("config/HiFi-GAN V3.json"))
    #settings = json.load(open("config/iSTFTNet.json"))
    #settings = json.load(open("config/MISRNet.json"))
    #settings = json.load(open("config/iSTFTMISRNet.json"))
    train_module = trainmodule.Vocoder(settings=settings)
    data_module = datamodule.VocoderDataModule(settings=settings)

    callbacks = [plc.ModelCheckpoint(), plc.RichProgressBar()]

    trainer = pl.Trainer(max_epochs=settings["learning_option"]["num_epochs"], callbacks=callbacks, precision="32")

    trainer.fit(train_module, data_module)

