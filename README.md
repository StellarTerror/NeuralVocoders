# Unofficial Vocoder Implements

## Implementations
- HiFi-GAN
- iSTFTNet
- MISRNet

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