import argparse
import json
import torch
from modules import mel, generators
import soundfile as sf
import resampy

def recon(config, model_path):
    data_option = config["data_option"]
    melspec = mel.Mel(filter_length=data_option["fft_length"], n_mels=data_option["num_mels"], sampling_rate=data_option["sample_rate"], hop_length=data_option["hop_length"], win_length=data_option["win_length"], fmin=data_option["fmin"], fmax=data_option["fmax"], center=False)
    generator = generators.Generator.load(model_path, config).cuda()

    sound, sr = sf.read("example.wav")
    sound = resampy.resample(sound, sr, data_option["sample_rate"])
    orig = melspec(torch.from_numpy(sound).unsqueeze(0))

    # generate audio
    with torch.no_grad():
        reconstructed = generator(orig.cuda())
    reconstructed = reconstructed.squeeze().cpu().detach().numpy()

    # save audio
    sf.write("reconstructed.wav", reconstructed, data_option["sample_rate"])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="config/HiFi-GAN V1.json")
    argparser.add_argument("--model", type=str, default="pretrained/HiFi-GAN V1.pt")

    args = argparser.parse_args()
    config = json.load(open(args.config, "r"))
    recon(config, args.model)


    