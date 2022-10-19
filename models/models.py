from typing import Type, Any, Callable, Union, List, Optional

import torch.nn as nn
from torch import Tensor

import segmentation_models_pytorch as SMP
from vit_pytorch import ViT
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

from .unet import Unet
from .axialnet import ResAxialAttentionUNet, medt_net, AxialBlock, AxialBlock_dynamic, AxialBlock_wopos


def deeplabv3plus_resnet50(args, **kwargs):
    """Constructs a DeepLabV3+ model with a ResNet-50 backbone.

    Args:
        in_channels (int):  A number of input channels for the model, default is 3 (RGB images)
        classes (int):      A number of classes for output mask 
                            (or you can think as a number of channels of output mask)
        encoder_name (str): Name of the classification model that will be used as an encoder (a.k.a
                            backbone) to extract features of different spatial resolution
        encoder_depth (int): A number of stages used in encoder in range [3, 5].
                            Each stage generate features two times smaller in spatial dimentions than previous one 
                            (e.g. for depth 0 we will have features with shapes [(N, C, H, W),], 
                            for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). (Default is 5)
        encoder_weights (str): One of None (random initialization), “imagenet” (pre-training on ImageNet) 
                            and other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride (int): Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates (tuple): Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels (int): A number of convolution filters in ASPP module. Default is 256
        activation (str):   An activation function to apply after the final convolution layer. Avaliable
                            options are “sigmoid”, “softmax”, “logsoftmax”, “identity”, callable and None. 
                            (Default is None)
        upsampling (int):   Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params (dict):  Dictionary with parameters of the auxiliary output (classification head).
                            Auxiliary output is build on top of encoder if aux_params is not None (default). Supported
                                params:
                                - classes (int): A number of classes
                                - pooling (str): One of “max”, “avg”. Default is “avg”
                                - dropout (float): Dropout factor in [0, 1)
                                - activation (str): An activation function to apply “sigmoid”/”softmax” 
                                (could be None to return logits)
    """
    encoder_name = 'resnet50'
    encoder_depth = args.encoder_depth
    encoder_weights = args.encoder_weights
    encoder_output_stride = args.encoder_output_stride
    decoder_atrous_rates = args.decoder_atrous_rates
    decoder_channels = args.decoder_channels
    in_channels = args.in_channels
    classes = args.classes
    activation = args.activation
    upsampling = args.upsampling
    aux_params = args.aux_params

    return SMP.DeepLabV3Plus(encoder_name=encoder_name, 
                            encoder_depth=encoder_depth, 
                            encoder_weights=encoder_weights, 
                            encoder_output_stride=encoder_output_stride, 
                            decoder_channels=decoder_channels, 
                            decoder_atrous_rates=decoder_atrous_rates, 
                            in_channels=in_channels, 
                            classes=classes, 
                            activation=activation, 
                            upsampling=upsampling, 
                            aux_params=aux_params )

def deeplabv3plus_resnet101(args, **kwargs):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone."""

    encoder_name = 'resnet101'
    encoder_depth = args.encoder_depth
    encoder_weights = args.encoder_weights
    encoder_output_stride = args.encoder_output_stride
    decoder_atrous_rates = args.decoder_atrous_rates
    decoder_channels = args.decoder_channels
    in_channels = args.in_channels
    classes = args.classes
    activation = args.activation
    upsampling = args.upsampling
    aux_params = args.aux_params

    return SMP.DeepLabV3Plus(encoder_name=encoder_name, 
                            encoder_depth=encoder_depth, 
                            encoder_weights=encoder_weights, 
                            encoder_output_stride=encoder_output_stride, 
                            decoder_channels=decoder_channels, 
                            decoder_atrous_rates=decoder_atrous_rates, 
                            in_channels=in_channels, 
                            classes=classes, 
                            activation=activation, 
                            upsampling=upsampling, 
                            aux_params=aux_params )

def unet(args, **kwargs, ):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation."""
    
    return Unet(n_channels=3, n_classes=2, )

"""
Medical Transformer: Gated Axial-Attention for Medical Image Segmentation

"""
def axialunet(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model

def gated(pretrained=False, **kwargs):
    model = ResAxialAttentionUNet(AxialBlock_dynamic, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model

def medt(args, pretrained=False, **kwargs):
    img_size = args.MedT_img_size
    model = medt_net(AxialBlock_dynamic,AxialBlock_wopos, [1, 2, 4, 1], s=0.125, img_size=img_size,  **kwargs)
    return model

def logo(pretrained=False, **kwargs):
    model = medt_net(AxialBlock,AxialBlock, [1, 2, 4, 1], s= 0.125, **kwargs)
    return model


def vit(args, **kwargs, ):

    return ViT(**kwargs)