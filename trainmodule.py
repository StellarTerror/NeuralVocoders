from modules.generators import Generator
from modules.discriminators import Discriminator
from modules.mel import Mel
import lightning.pytorch as pl
import torch
from modules.mel import Mel
import os
import matplotlib.pyplot as plt
from copy import deepcopy

class Vocoder(pl.LightningModule):
    def __init__(self, config):
        super(Vocoder, self).__init__()
        self.config = config
        self.data_config = self.config["data_option"]
        self.learning_config = self.config["learning_option"]
        self.model_config = self.config["model"]

        self.mel = Mel(filter_length=self.data_config["fft_length"], sampling_rate=self.data_config["sample_rate"], n_mels=self.data_config["num_mels"], hop_length=self.data_config["hop_length"], 
                    win_length=self.data_config["win_length"], fmin=self.data_config["fmin"], fmax=self.data_config["fmax"])
        
        self.generator = Generator(self.config)
        self.discreminator = Discriminator(self.mel)

        self.mel = self.mel.eval()
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
    
    def training_step(self, batch, _):
        g_opt, d_opt = self.optimizers()

        wave, _ = batch

        with torch.no_grad():
            x = self.mel(wave)
            y = wave
            y_mel = deepcopy(x)

        x = torch.autograd.Variable(x.to(self.device, non_blocking=True))
        y = torch.autograd.Variable(y.to(self.device, non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to(self.device, non_blocking=True))
        y = y.unsqueeze(1)

        # Discriminator Training
        y_g_hat = self.generator(x)
        
        loss = self.discreminator(y, y_g_hat, gen_train=False)

        d_opt.zero_grad()
        self.manual_backward(loss)
        d_opt.step()

        # Generator Training
        y_g_hat = self.generator(x)
        loss, loss_mel = self.discreminator(y, y_g_hat, gen_train=True)

        self.log("Gen Loss Total", loss, prog_bar=True)
        self.log("Mel-Spec Error", loss_mel, prog_bar=True)

        g_opt.zero_grad()
        self.manual_backward(loss)
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

    def validation_step(self, batch, _):
        if (len(self.recon) > 5 and len(self.orignal) > 5):
            return
        
        wave, _ = batch

        with torch.no_grad():
            x = self.mel(wave)
            y = wave
            y_mel = deepcopy(x)

        x = torch.autograd.Variable(x.to(self.device, non_blocking=True))
        y = torch.autograd.Variable(y.to(self.device, non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to(self.device, non_blocking=True))
        y = y.unsqueeze(1)

        y_recon = self.generator(x)
        y_original = y

        self.orignal.append(y_original.squeeze())
        self.recon.append(y_recon.squeeze())

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        for i in range(len(self.orignal)):
            self.logger.experiment.add_audio(f"Original/{i}", self.orignal[i], self.current_epoch, sample_rate=self.data_config["sample_rate"])
            self.logger.experiment.add_audio(f"Reconstructed/{i}", self.recon[i], self.current_epoch, sample_rate=self.data_config["sample_rate"])

            y_orig_mel = self.mel(self.orignal[i])
            y_recon_mel = self.mel(self.recon[i])

            fig, ax = plt.subplots(1, 1)
            ax.imshow(y_orig_mel.cpu().detach().numpy(), aspect='auto', origin='lower')
            ax.set_title("Original Mel-Spectrogram")
            self.logger.experiment.add_figure(f"Original Mel-Spectrogram/{i}", fig, self.current_epoch)
            plt.close(fig)

            fig, ax = plt.subplots(1, 1)
            ax.imshow(y_recon_mel.cpu().detach().numpy(), aspect='auto', origin='lower')
            ax.set_title("Reconstructed Mel-Spectrogram")
            self.logger.experiment.add_figure(f"Reconstructed Mel-Spectrogram/{i}", fig, self.current_epoch)
            plt.close(fig)

        self.orignal.clear()
        self.recon.clear()

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(self.generator.parameters(), self.learning_config["lr"], betas=(self.learning_config["betas"][0], self.learning_config["betas"][1]))
        d_opt = torch.optim.AdamW(self.discreminator.parameters(), self.learning_config["lr"], betas=(self.learning_config["betas"][0], self.learning_config["betas"][1]))
        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_opt, gamma=self.learning_config["lr_decay"])
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_opt, gamma=self.learning_config["lr_decay"])
        return [g_opt, d_opt], [g_scheduler, d_scheduler]