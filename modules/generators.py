import torch
import torch.nn.functional as F
import torch.nn as nn

from .stft import ISTFT

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class MRF(nn.Module):
    def __init__(self, in_channel, resblock_kernel_sizes, resblock_dilation_sizes):
        super(MRF, self).__init__()
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes

        self.resblocks = nn.ModuleList()
        for i in range(len(resblock_kernel_sizes)):
            self.resblocks.append(ResBlock(in_channel, in_channel, resblock_kernel_sizes[i], resblock_dilation_sizes[i]))

    def forward(self, x):
        blocks = []
        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x)
            x = F.mish(x)
            blocks.append(x)
        x = sum(blocks)/len(blocks)  
        return x

class MISR(nn.Module):
    def __init__(self, in_channel, resblock_kernel_size, resblock_dilation_sizes):
        super(MISR, self).__init__()
        self.resblock_kernel_sizes = resblock_kernel_size
        self.resblock_dilation_sizes = resblock_dilation_sizes

        self.in_conv = nn.ModuleList()
        for _ in range(len(resblock_dilation_sizes)):
            self.in_conv.append(nn.Conv1d(in_channel, in_channel, 1, 1, padding=0))
        self.out_conv = nn.Conv1d(in_channel * 3, in_channel, 1, 1, padding=0)
        self.resblock = ResBlock(in_channel, in_channel, resblock_kernel_size, resblock_dilation_sizes)

    def forward(self, x):
        blocks = []
        for i in range(len(self.in_conv)):
            blocks.append(self.resblock(self.in_conv[i](x)))
        x = torch.cat(blocks, dim=1)
        x = self.out_conv(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.conv1s = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=get_padding(kernel_size, dilation[j]), dilation=dilation[j]) for j in range(len(dilation))
        ])
        self.conv2s = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=get_padding(kernel_size, 1), dilation=1) for j in range(len(dilation))
        ])

    def forward(self, x):
        resblocks = []
        for i in range(len(self.conv1s)):
            resblocks.append(F.mish(self.conv2s[i](F.mish(self.conv1s[i](x)))))
        return sum(resblocks)/len(resblocks)


class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_rate, kernel_size):
        super(Upsampler, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, \
                                        kernel_size, upsample_rate, padding=(kernel_size - upsample_rate)//2)
        
    def forward(self, x):
        return self.upsample(x)

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        model_config = config["model"]
        self.upsample_num = len(model_config["upsample_rate"])
        

        self.upsampler = nn.ModuleList([
            Upsampler(model_config["upsample_initial_channel"]//(2**k), \
                      model_config["upsample_initial_channel"]//(2**(k+1)),
                       model_config["upsample_rate"][k], model_config["upsample_kernel_size"][k]) for k in range(self.upsample_num)
        ])

        if model_config["resblock_type"] == "MRF":
            self.mrf = nn.ModuleList([MRF(model_config["upsample_initial_channel"]//(2**(k+1)), model_config["resblock_kernel_sizes"], model_config["resblock_dilation_sizes"]) for k in range(self.upsample_num)])
        elif model_config["resblock_type"] == "MISR":
            self.misr = nn.ModuleList([MISR(model_config["upsample_initial_channel"]//(2**(k+1)), model_config["resblock_kernel_sizes"], model_config["resblock_dilation_sizes"]) for k in range(self.upsample_num)])

        self.first_conv = nn.Conv1d(80, model_config["upsample_initial_channel"], 7, 1, padding=3)


        if model_config["istft_use"] == True:
            self.n_fft = model_config["gen_istft_n_fft"]
            self.istft = ISTFT(filter_length=model_config["gen_istft_n_fft"], hop_length=model_config["gen_istft_hop_size"], win_length=model_config["gen_istft_n_fft"], window='hann')
            self.final_conv = nn.Conv1d(model_config["upsample_initial_channel"]//(2 ** self.upsample_num), model_config["gen_istft_n_fft"] + 2, 7, 1, padding=3)
            self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        else:
            self.final_conv = nn.Conv1d(model_config["upsample_initial_channel"] // (2 ** self.upsample_num), 1, 7, 1, padding=3)
            self.activation = nn.Tanh()

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path, config):
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model

    def forward(self, x):
        x = self.first_conv(x)
        for k in range(len(self.upsampler)):
            x = F.mish(x)
            x = self.upsampler[k](x)
            if hasattr(self, 'mrf'):
                x = self.mrf[k](x)
            elif hasattr(self, 'misr'):
                x = self.misr[k](x)
        if hasattr(self, 'istft'):
            x = self.reflection_pad(x)
            x = self.final_conv(x)
            magnitude = torch.exp(x[:,:(self.n_fft) // 2 + 1, :])
            phase = torch.sin(x[:, (self.n_fft)// 2 + 1:, :])
            x = self.istft(magnitude, phase)
        else:
            x = self.final_conv(x)
            x = self.activation(x)
        return x