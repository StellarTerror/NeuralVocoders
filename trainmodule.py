from modules.generators import Generator
from modules.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator,\
    discriminator_loss, feature_loss, generator_loss
from modules.mel import Mel
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from modules.mel import Mel
import os, shutil

import matplotlib.pyplot as plt

class Vocoder(pl.LightningModule):
    def __init__(self, config):
        super(Vocoder, self).__init__()
        self.config = config
        self.data_config = self.config["data_option"]
        self.learning_config = self.config["learning_option"]
        self.model_config = self.config["model"]

        self.generator = Generator(self.config)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        self.mpd = torch.compile(self.mpd, mode="reduce-overhead")
        self.msd = torch.compile(self.msd, mode="reduce-overhead")

        self.mel = Mel(filter_length=self.data_config["fft_length"], sampling_rate=self.data_config["sample_rate"], n_mels=self.data_config["num_mels"], hop_length=self.data_config["hop_length"], win_length=self.data_config["win_length"], fmin=self.data_config["fmin"], fmax=self.data_config["fmax"])
        for param in self.mel.parameters():
            param.requires_grad = False

        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints", exist_ok=True)

        self.automatic_optimization = False

    def forward(self, x):
        self.generator.eval()
        with torch.no_grad():
            y_hat = self.generator(x)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        x, y, _, y_mel = batch
        x = torch.autograd.Variable(x.to(self.device, non_blocking=True))
        y = torch.autograd.Variable(y.to(self.device, non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to(self.device, non_blocking=True))
        y = y.unsqueeze(1)

        # Discriminator Training
        y_g_hat = self.generator(x)
        y_g_hat_mel = self.mel(y_g_hat)

        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        d_opt.zero_grad()
        self.manual_backward(loss_disc_all)
        d_opt.step()

        # Generator Training
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        with torch.no_grad():
            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
        self.log("Gen Loss Total", loss_gen_all, prog_bar=True)
        self.log("Mel-Spec Error", mel_error, prog_bar=True)

        g_opt.zero_grad()
        self.manual_backward(loss_gen_all)
        g_opt.step()

    def on_train_epoch_end(self) -> None:
        lrschedules = self.lr_schedulers()
        for lrschedule in lrschedules:
            lrschedule.step()

        if os.path.exists(f"checkpoints/{self.model_config['name']}_{self.current_epoch-1}.pt"):
            os.remove(f"checkpoints/{self.model_config['name']}_{self.current_epoch-1}.pt")
        self.generator.save(f"checkpoints/{self.model_config['name']}_{self.current_epoch}.pt")

    def on_validation_epoch_start(self) -> None:
        self.orignal = []
        self.recon = []

    def validation_step(self, batch, batch_idx):
        if (len(self.recon) > 5 and len(self.orignal) > 5):
            return
        x, y, _, y_mel = batch
        x = torch.autograd.Variable(x.to(self.device, non_blocking=True))
        y = torch.autograd.Variable(y.to(self.device, non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to(self.device, non_blocking=True))
        y = y.unsqueeze(1)

        y_recon = self.generator(x)
        y_original = y

        self.orignal.append(y_original)
        self.recon.append(y_recon)

    
    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        for i in range(5):
            self.logger.experiment.add_audio(f"Original/{i}", self.orignal[i], self.current_epoch, sample_rate=self.data_config["sample_rate"])
            self.logger.experiment.add_audio(f"Reconstructed/{i}", self.recon[i], self.current_epoch, sample_rate=self.data_config["sample_rate"])

            y_orig_mel = self.mel(self.orignal[i])
            y_recon_mel = self.mel(self.recon[i])

            fig, ax = plt.subplots(1, 1)
            ax.imshow(y_orig_mel[0].cpu().detach().numpy(), aspect='auto', origin='lower')
            ax.set_title("Original Mel-Spectrogram")
            self.logger.experiment.add_figure(f"Original Mel-Spectrogram/{i}", fig, self.current_epoch)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(y_recon_mel[0].cpu().detach().numpy(), aspect='auto', origin='lower')
            ax.set_title("Reconstructed Mel-Spectrogram")
            self.logger.experiment.add_figure(f"Reconstructed Mel-Spectrogram/{i}", fig, self.current_epoch)
            plt.close(fig)

        self.orignal.clear()
        self.recon.clear()

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(self.generator.parameters(), self.learning_config["lr"], betas=(self.learning_config["betas"][0], self.learning_config["betas"][1]))
        d_opt = torch.optim.AdamW(chain(self.msd.parameters(), self.mpd.parameters()), self.learning_config["lr"], betas=(self.learning_config["betas"][0], self.learning_config["betas"][1]))
        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_opt, gamma=self.learning_config["lr_decay"])
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_opt, gamma=self.learning_config["lr_decay"])
        return [g_opt, d_opt], [g_scheduler, d_scheduler]