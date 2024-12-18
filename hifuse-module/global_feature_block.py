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

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class ShiftedWindowAttention(nn.Module):
   

    def __init__(self, input_dim, win_size, num_heads, include_bias=True, attention_dropout=0., projection_dropout=0.):
        super().__init__()
        self.input_dim = input_dim
        self.win_size = win_size  # (height, width)
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scaling_factor = head_dim ** -0.5

        # Initialize a parameter table for relative position encoding
        self.relative_position_encoding = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads)
        )

        # Compute relative positional indices
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # Shape: [2, height, width]
        coords_flattened = coords.flatten(1)  # Flatten along spatial dimensions: [2, height*width]
        relative_coords = coords_flattened[:, :, None] - coords_flattened[:, None, :]  # [2, height*width, height*width]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Shape: [height*width, height*width, 2]
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        position_index = relative_coords.sum(-1)  # Shape: [height*width, height*width]
        self.register_buffer("relative_position_index", position_index)

        # Attention components
        self.qkv_projection = nn.Linear(input_dim, input_dim * 3, bias=include_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Linear(input_dim, input_dim)
        self.output_dropout = nn.Dropout(projection_dropout)

        nn.init.trunc_normal_(self.relative_position_encoding, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Compute the attention for the given input.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size*num_windows, window_height*window_width, input_dim).
            mask (Optional[Tensor]): Mask tensor with shape (num_windows, window_size*window_size, window_size*window_size), or None.

        Returns:
            Tensor: The output tensor after applying window-based attention.
        """
        B_, N, C = x.shape

        # Compute query, key, and value projections
        qkv = self.qkv_projection(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, batch_size*num_windows, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Scale the query
        q = q * self.scaling_factor
        attention_scores = q @ k.transpose(-2, -1)  # Compute attention scores: [batch_size*num_windows, num_heads, N, N]

        # Add relative position bias
        relative_bias = self.relative_position_encoding[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1
        )
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attention_scores += relative_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention_scores = attention_scores.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attention_scores += mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, N, N)
            attention_probs = self.softmax(attention_scores)
        else:
            attention_probs = self.softmax(attention_scores)
        
        attention_probs = self.attention_dropout(attention_probs)

        # Compute attention output
        context = (attention_probs @ v).transpose(1, 2).reshape(B_, N, C)
        context = self.output_projection(context)
        context = self.output_dropout(context)

        return context
    
class GlobalFeatureExtractor(nn.Module):

    def __init__ (self, channels, attention_heads, win_size=7, shift_val=0, use_bias=True, dropout_rate=0., attn_dropout=0., stochastic_depth=0., activation_func=nn.GELU, normalization_layer=nn.LayerNorm ):
        super().__init__()
        self.input_dim = channels
        self.num_attention_heads = attention_heads
        self.win_size = win_size
        self.shift_val = shift_val
        assert 0 <= self.shift_val < self.win_size, "shift_val must be in the range [0, win_size)"

        self.norm1 = nn.LayerNorm(channels)
        self.shifted_window_msa = ShiftedWindowAttention( channels, win_size=(self.win_size, self.win_size), num_heads=attention_heads, 
            include_bias=use_bias, attention_dropout=attn_dropout, projection_dropout=dropout_rate)

        self.stochastic_depth = DropPath(stochastic_depth) if  stochastic_depth > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(channels)
        self.linear_layer = nn.Linear(channels, channels)
        self.activation = activation_func()

    def forward(self, inputs, attention_mask):
        height, width = self.H, self.W
        batch_size, length, channels = inputs.shape
        assert length == height * width, "Input dimensions mismatch."

        shortcut = inputs
        inputs = self.norm1(inputs)
        inputs = inputs.view(batch_size, height, width, channels)

        # Padding
        padding_left = padding_top = 0
        padding_right = (self.win_size - width % self.win_size) % self.win_size
        padding_bottom = (self.win_size - height % self.win_size) % self.win_size
        inputs = F.pad(inputs, (0, 0, padding_left, padding_right, padding_top, padding_bottom))
        padded_height, padded_width, _ = inputs.shape[1:]

        # Apply cyclic shift
        if self.shift_val > 0:
            shifted_inputs = torch.roll(inputs, shifts=(-self.shift_val, -self.shift_val), dims=(1, 2))
        else:
            shifted_inputs = inputs
            attention_mask = None

        # Partition and process windows
        windows = window_partition(shifted_inputs, self.win_size)
        windows = windows.view(-1, self.win_size * self.win_size, channels)

        # Apply attention
        processed_windows = self.shifted_window_msa(windows, mask=attention_mask)
        processed_windows = processed_windows.view(-1, self.win_size, self.win_size, channels)

        # Merge windows back
        shifted_inputs = window_reverse(processed_windows, self.win_size, padded_height, padded_width)

        # Reverse the cyclic shift
        if self.shift_val > 0:
            inputs = torch.roll(shifted_inputs, shifts=(self.shift_val, self.shift_val), dims=(1, 2))
        else:
            inputs = shifted_inputs

        # Remove padding
        if padding_right > 0 or padding_bottom > 0:
            inputs = inputs[:, :height, :width, :].contiguous()

        inputs = inputs.view(batch_size, height * width, channels)
        inputs = self.linear_layer(inputs)
        inputs = self.activation(inputs)
        inputs = shortcut + self.stochastic_depth(inputs)

        return inputs
    
class BasicLayer(nn.Module):
    """
    Downsampling and Global Feature Block for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            GlobalFeatureExtractor(
                channels=dim,
                attention_heads=num_heads,
                win_size=window_size,
                shift_val=0 if (i % 2 == 0) else self.shift_size,
                use_bias=qkv_bias,
                dropout_rate=drop,
                attn_dropout=attn_drop,
                stochastic_depth=drop_path[i] if isinstance(drop_path, list) else drop_path,
                normalization_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):

        if self.downsample is not None:
            x = self.downsample(x, H, W)         #patch merging stage2 in [6,3136,96] out [6,784,192]
            H, W = (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:                  # global block
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, H, W
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim//2
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x
    
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # downsample patch_size times
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W
    

def show_embedded_patches(patches, title="Embedded Patches"):
    """
    Visualize linear-embedded patches by averaging over the embedding dimension.
    The output will be a grayscale representation of the patches.
    """
    B, N, C = patches.shape  #
    grid_size = int(np.sqrt(N))  
    patches = patches.view(B, grid_size, grid_size, C)
    
    patches_mean = patches.mean(dim=-1) 
    patches_np = patches_mean.squeeze(0).cpu().detach().numpy() 
    min_val, max_val = patches_np.min(), patches_np.max()
    patches_np = (patches_np - min_val) / (max_val - min_val)  
    
    # Display the grid of patches
    plt.imshow(patches_np, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
