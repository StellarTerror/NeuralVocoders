# Unofficial Vocoder Implementations

## Implemented Vocoder List
- HiFi-GAN
- iSTFTNet
- MISRNet
- and more

## Exporting

| Export Method  | HiFi-GAN | iSTFTNet | MISRNet | 
| -------------- | -------- | -------- | ------- | 
| onnx           | ⭕       | ❌       | ⭕      | 
| pytorch.export | ⭕       | ❌       | ⭕      | 
| state_dict     | ⭕       | ⭕       | ⭕      | 

## How to Train

```
$ python train.py --config "config/HiFi-GAN V1.json"
```

## How to Test

```
$ python recon.py --config "config/HiFi-GAN V1.json" --model "pretrained/HiFi-GAN V1.pt"
```
## Pretrained Model

You can download by below link

[Download](https://1drv.ms/f/s!Al49QwC7fKujlyrTwJYTVdPIY0vk?e=eCOhls)

## Acknowledgements
I referred to [hifi-gan](https://github.com/jik876/hifi-gan), [iSTFTNet](https://arxiv.org/pdf/2203.02395.pdf), and [MISRNet](https://www.isca-archive.org/interspeech_2022/kaneko22_interspeech.pdf) to implement this.