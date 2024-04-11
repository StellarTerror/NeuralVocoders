import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn

class Mel(nn.Module):
    def __init__(self, filter_length, n_mels, sampling_rate, hop_length, win_length, fmin, fmax, center=False):
        super(Mel, self).__init__()
        
        self.mel_basis = torch.from_numpy(librosa_mel_fn(sr = sampling_rate, n_fft=filter_length, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False, norm='slaney')).float()
        self.hann_window = torch.hann_window(win_length)

        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.center = center

    def forward(self, y):
        self.hann_window = self.hann_window.to(y.device)
        self.mel_basis = self.mel_basis.to(y.device)
        
        if y.dim() == 1:
            y = y.unsqueeze(0)
            y = torch.nn.functional.pad(y, (int((self.n_fft-self.hop_size)/2), int((self.n_fft-self.hop_size)/2)), mode='reflect')
            y = y.squeeze(0)
        else:
            y = torch.nn.functional.pad(y, (int((self.n_fft-self.hop_size)/2), int((self.n_fft-self.hop_size)/2)), mode='reflect')
            y = y.squeeze(1)
        
        spec = torch.stft(y, self.n_fft, self.hop_size, self.win_size, window=self.hann_window, center=self.center, return_complex=True)
        spec = torch.stack([spec.real, spec.imag], -1)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1)

        return spec