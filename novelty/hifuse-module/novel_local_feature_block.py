import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

class LayerNormChannelsFirst(nn.Module):
    """
    LayerNorm for inputs with shape (batch_size, channels, height, width), i.e., channels_first format.
    
    Args:
        normalized_shape (int or list): The shape of the input to normalize (usually the number of channels).
        eps (float): A small constant for numerical stability. Default is 1e-6.
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNormChannelsFirst, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom implementation for layer normalization in channels_first format
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # Apply the learnable weight and bias
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class LayerNormChannelsLast(nn.Module):
    """
    LayerNorm for inputs with shape (batch_size, height, width, channels), i.e., channels_last format.
    Args:
        normalized_shape (int or list): The shape of the input to normalize (usually the number of channels).
        eps (float): A small constant for numerical stability. Default is 1e-6.
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNormChannelsLast, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's built-in layer_norm function for channels_last format
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    


class Convolution_2d_k4_s4(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, kernel_size=4, stride=4):
        super(Convolution_2d_k4_s4, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LocalFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalFeatureBlock, self).__init__()
        self.depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm1 = LayerNormChannelsFirst(out_channels)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_separable_conv(x)
        x = self.relu(x) 
        x = self.norm1(x)
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.norm1(x)
        return x

class LocalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(LocalFeatureExtractor, self).__init__()
        self.local_feature_block = LocalFeatureBlock(in_channels, out_channels)
        self.stochastic_depth = nn.DropPath(dropout_prob) if dropout_prob > 0. else nn.Identity()

    def forward(self, x):
        x = self.local_feature_block(x)
        output = self.stochastic_depth(x)
        return output
    