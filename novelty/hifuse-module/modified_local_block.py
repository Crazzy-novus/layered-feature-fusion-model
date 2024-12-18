import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
import torch.utils.checkpoint as checkpoint


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


class LocalFeatureWeightedFusionMLP(nn.Module):
    def __init__(self, channels, dropout_prob=0.0):
        super(LocalFeatureWeightedFusionMLP, self).__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.layer_norm = LayerNormChannelsLast(channels, eps=1e-6)
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)  # Pointwise convolution
        self.activation = nn.GELU()  # Activation function
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0. else nn.Identity()
        
        # Additional parameters for the shift operation
        self.shift_size = 1  # Define the shift size (can be adjusted)
        
    def shift(self, x):
        # Shift operation to exchange information between local features
        N, C, H, W = x.size()
        # Shift right
        x_shifted_right = torch.roll(x, shifts=self.shift_size, dims=3)
        # Shift down
        x_shifted_down = torch.roll(x, shifts=self.shift_size, dims=2)
        
        # Combine the original and shifted features
        return x + x_shifted_right + x_shifted_down

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        residual = input_tensor  # Save the input for residual connection
        x = self.depthwise_conv(input_tensor)
        x = self.layer_norm(x)
        x = self.shift(x)  # Apply the shift operation
        x = self.pointwise_conv(x)
        x = self.activation(x)
        x = self.dropout(x)  # Apply dropout if specified
        output = residual + x  # Add residual connection
        return output