import torch
import torch.nn as nn
from global_feature_block import DropPath
from local_feature__block import LayerNormChannelsFirst, LayerNormChannelsLast


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, use_bn=False, use_relu=True, use_bias=True, groups=1):
        super(ConvolutionLayer, self).__init__()
        
        self.input_channels = input_channels
        
        # Define the convolution layer
        self.convolution = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size, 
            stride, 
            padding=(kernel_size - 1) // 2, 
            bias=use_bias, 
            groups=groups
        )
        
        # Conditional addition of ReLU activation
        self.activation = nn.ReLU(inplace=True) if use_relu else None
        
        # Conditional addition of Batch Normalization
        self.batch_norm = nn.BatchNorm2d(output_channels) if use_bn else None

    def forward(self, x):
        # Ensure input channel size matches
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected input channels {self.input_channels}, but got {x.size(1)}")
        
        # Apply convolution
        x = self.convolution(x)
        
        # Apply batch normalization if defined
        if self.batch_norm:
            x = self.batch_norm(x)
        
        # Apply activation if defined
        if self.activation:
            x = self.activation(x)
        
        return x


class Hifuse_block (nn.Module):
    def __init__(self, input_channels, mid_channels, reduction_factor, intermediate_channels, output_channels, dropout_rate=0.0):
        super(Hifuse_block, self).__init__()
        
        # Pooling layers
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Squeeze and Excitation block
        self.squeeze_excite = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // reduction_factor, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels // reduction_factor, mid_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.Sigmoid()
        self.spatial_conv = ConvolutionLayer(2, 1, kernel_size=7, use_bn=True, use_relu=False, use_bias=False)
        self.local_transform = ConvolutionLayer(input_channels, intermediate_channels, kernel_size=1, use_bn=True, use_relu=False)
        self.global_transform = ConvolutionLayer(mid_channels, intermediate_channels, kernel_size=1, use_bn=True, use_relu=False)
        self.average_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dim_upscale = ConvolutionLayer(intermediate_channels // 2, intermediate_channels, kernel_size=1, use_bn=True, use_relu=True)

        self.layer_norm1 = LayerNormChannelsFirst(intermediate_channels * 3, eps=1e-6)
        self.layer_norm2 = LayerNormChannelsFirst(intermediate_channels * 2, eps=1e-6)
        self.layer_norm3 = LayerNormChannelsFirst(input_channels + mid_channels + intermediate_channels, eps=1e-6)

        self.weight_transform1 = ConvolutionLayer(intermediate_channels * 3, intermediate_channels, kernel_size=1, use_bn=True, use_relu=False)
        self.weight_transform2 = ConvolutionLayer(intermediate_channels * 2, intermediate_channels, kernel_size=1, use_bn=True, use_relu=False)

        self.non_linear_activation = nn.GELU()

        self.residual_block = IRMLP(input_channels + mid_channels + intermediate_channels, output_channels)
        self.drop_path_layer = DropPath(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, local_features, global_features, transformer_features):
        # Transform local features using the local feature block
        transformed_local = self.local_transform(local_features)
        
        # Transform global features using the global feature block
        transformed_global = self.global_transform(global_features)
        
        # Process transformer features if available
        if transformer_features is not None:
            upsampled_transformer = self.dim_upscale(transformer_features)
            pooled_transformer = self.average_pool(upsampled_transformer)
            shortcut = pooled_transformer
            
            # Concatenate transformer, local, and global features
            fused_features = torch.cat([pooled_transformer, transformed_local, transformed_global], dim=1)
            fused_features = self.layer_norm1(fused_features)
            fused_features = self.weight_transform1(fused_features)
            fused_features = self.non_linear_activation(fused_features)
        else:
            shortcut = 0
            
            # Concatenate local and global features
            fused_features = torch.cat([transformed_local, transformed_global], dim=1)
            fused_features = self.layer_norm2(fused_features)
            fused_features = self.weight_transform2(fused_features)
            fused_features = self.non_linear_activation(fused_features)

        # Spatial attention for ConvNeXt branch
        local_jump = local_features
        max_local, _ = torch.max(local_features, dim=1, keepdim=True)
        avg_local = torch.mean(local_features, dim=1, keepdim=True)
        spatial_attention_input = torch.cat([max_local, avg_local], dim=1)
        spatial_attention = self.spatial_conv(spatial_attention_input)
        spatial_attention = self.activation(spatial_attention) * local_jump

        # Channel attention for transformer branch
        global_jump = global_features
        max_global = self.max_pool(global_features)
        avg_global = self.avg_pool(global_features)
        max_attention = self.squeeze_excite(max_global)
        avg_attention = self.squeeze_excite(avg_global)
        channel_attention = self.activation(max_attention + avg_attention) * global_jump

        # Fuse all branches and apply the residual block
        fused_output = torch.cat([channel_attention, spatial_attention, fused_features], dim=1)
        fused_output = self.layer_norm3(fused_output)
        fused_output = self.residual_block(fused_output)
        fused_output = shortcut + self.drop_path_layer(fused_output)

        return fused_output
    
#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = ConvolutionLayer(inp_dim, inp_dim, 3, use_relu=False, use_bias=False, groups=inp_dim)
        self.conv2 = ConvolutionLayer(inp_dim, inp_dim * 4, 1, use_relu=False, use_bias=False)
        self.conv3 = ConvolutionLayer(inp_dim * 4, out_dim, 1, use_relu=False, use_bias=False, use_bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out
