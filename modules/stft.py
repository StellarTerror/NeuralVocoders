import torch
import numpy as np
from scipy.signal import get_window

class STFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def forward(self, input_data):
        window = self.window.to(input_data.device)
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)


class ISTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))
    
    def forward(self, magnitude, phase):
        window = self.window.to(magnitude.device)
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=window)

        return inverse_transform.unsqueeze(-2)