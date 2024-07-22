import torch
import torch.nn as nn
import torch.nn.functional as F
from .generators import get_padding

class Discriminator(torch.nn.Module):
    def __init__(self, mel):
        super(Discriminator, self).__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.msd = torch.compile(self.msd, backend="inductor", mode="reduce-overhead")
        self.mpd = torch.compile(self.mpd, backend="inductor", mode="reduce-overhead")
        self.mel = mel.eval()
        for param in self.mel.parameters():
            param.requires_grad = False

    def _feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss*2

    def _discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


    def _generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
    
    def forward(self, y, y_hat, gen_train = False):
        y_mel = self.mel(y)
        y_mel_hat = self.mel(y_hat)

        if gen_train:
            # generator_train
            y_df_hat_r, y_df_hat_g, y_fmap_r, y_fmap_g = self.mpd(y, y_hat)
            y_ds_hat_r, y_ds_hat_g, y_smap_r, y_smap_g = self.msd(y, y_hat)

            loss_mel = F.l1_loss(y_mel, y_mel_hat) * 45
            loss_fm_f = self._feature_loss(y_fmap_r, y_fmap_g)
            loss_fm_s = self._feature_loss(y_smap_r, y_smap_g)
            loss_gen_f, _ = self._generator_loss(y_df_hat_g)
            loss_gen_s, _ = self._generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            return loss_gen_all, loss_mel
        
        else:
            # discriminator_train
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat.detach())
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat.detach())

            loss_disc_f, _, _ = self._discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, _, _ = self._discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            return loss_disc_all
        
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.mish(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2)
        ])
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.mish(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
