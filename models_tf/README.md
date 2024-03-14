# Download models needed

Models we need the experiment include normal-trained models and advanced defense models. Both of them are available in https://github.com/tensorflow/models/tree/master/research/slim for normal-trained models and https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models for advanced defense models.

## How to download models on ImageNet dataset

Python script `download_models.py` allow you to download provided models.

Models mentioned above include: Inception-v3, Inception-v4, Inception-ResNet-v2, ResNet-v1 50, ResNet-v1 152, ResNet-v2 50, ResNet-v2 152, VGG-16, VGG-19, adv-inception-v3, ens3_adv_inception-v3, ens4_adv_inception-v3, adv_inception-ResNet-v2, ens3_adv_Inception-ResNet-v2.

Usage is as followed:

```shell
python download_moedls.py
```

