from modules.mel import Mel
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from glob import glob
import librosa
import numpy as np

import torch

import random

class VocoderDataModule(LightningDataModule):
    def __init__(self, settings):
        super(VocoderDataModule, self).__init__()
        self.settings = settings
        self.data_option = settings["data_option"]
        self.learning_option = settings["learning_option"]

        self.all_files = glob(self.data_option["audio_path"] + "/**/*.wav", recursive=True)
        random.shuffle(self.all_files)
        self.files = {"train": [], "val": []}
        
        self.files["train"] = self.all_files[:-5]
        self.files["val"] = self.all_files[-5:]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = VocoderDataset(self.settings, self.files["train"], is_train=True)
            self.val_dataset = VocoderDataset(self.settings, self.files["val"], is_train=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.learning_option["batch_size"], num_workers=self.learning_option["num_workers"], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.learning_option["num_workers"], shuffle=False)
    
class VocoderDataset(Dataset):
    def __init__(self, settings, audio_files, is_train=True):
        self.settings = settings
        self.audio_files = audio_files
        self.is_train = is_train
        self.data_option = settings["data_option"]
        self.mel = Mel(filter_length=self.data_option["fft_length"], sampling_rate=self.data_option["sample_rate"], n_mels=self.data_option["num_mels"], hop_length=self.data_option["hop_length"], win_length=self.data_option["win_length"], fmin=self.data_option["fmin"], fmax=self.data_option["fmax"])
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio, _ = librosa.load(self.audio_files[idx], sr=self.data_option["sample_rate"], mono=True)

        if self.is_train:
            if audio.shape[0] >= self.data_option["segment_length"]:
                max_audio_start = audio.shape[0] - self.data_option["segment_length"]
                audio_start = np.random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start+self.data_option["segment_length"]]
            else:
                audio = np.pad(audio, (0, self.data_option["segment_length"] - audio.shape[0]), mode="constant")
        
        audio = torch.from_numpy(audio)

        mel = self.mel(audio)
        mel_loss = self.mel(audio)

        return (mel.squeeze(), audio.squeeze(), self.audio_files[idx], mel_loss.squeeze())
