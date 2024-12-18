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
from global_feature_block import BasicLayer, PatchEmbed, PatchMerging
# from local_feature__block import LayerNormChannelsFirst, LocalFeatureExtractor, Convolution_2d_k4_s4
from novel_local_feature_block import LocalFeatureExtractor, Convolution_2d_k4_s4, LayerNormChannelsFirst
from utils import load_image, tensor_to_channel_images, visualize_channel_images
from feature_fussion_block import Hifuse_block

IMG_SIZE = 224
PATCH_SIZE = 4
NUM_CHANNELS = 96 


class HifuseModel(nn.Module):
    def __init__(self, num_classes, patch_size=4, in_chans=3, embed_dim=96, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, qkv_bias=True, drop_rate=0,
                 attn_drop_rate=0, drop_path_rate=0., norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, HFF_dp=0.,
                 conv_depths=(2, 2, 2, 2), conv_dims=(96, 192, 384, 768), conv_drop_path_rate=0.,
                 conv_head_init_scale: float = 1., **kwargs):
        super().__init__()


        ##### Local Branch Setting #######

        # Stem Layer
        self.conv_layer_0 =  Convolution_2d_k4_s4(in_channels=3, out_channels=96, kernel_size=4, stride=4)

        self.local_block_1 = LocalFeatureExtractor(in_channels=96, out_channels=96, dropout_prob=0.0)

        self.local_block_2 = LocalFeatureExtractor(in_channels=192, out_channels=192, dropout_prob=0.0)
        self.conv_layer_2 =  Convolution_2d_k4_s4(in_channels=96, out_channels=192, kernel_size=2, stride=2)

        self.local_block_3 = LocalFeatureExtractor(in_channels=384, out_channels=384, dropout_prob=0.0)
        self.conv_layer_3 =  Convolution_2d_k4_s4(in_channels=192, out_channels=384, kernel_size=2, stride=2)

        self.local_block_4 = LocalFeatureExtractor(in_channels=768, out_channels=768, dropout_prob=0.0)
        self.conv_layer_4 =  Convolution_2d_k4_s4(in_channels=384, out_channels=768, kernel_size=2, stride=2)



        # Built Stack

        self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)   # Modified to check train.py is running conv_dims[-1] -> 96
        # self.conv_norm = nn.LayerNorm(96, eps=1e-6)   # Modified to check train.py is running conv_dims[-1] -> 96
        self.conv_head = nn.Linear(conv_dims[-1], num_classes)
        # self.conv_head = nn.Linear(96, num_classes)
        self.conv_head.weight.data.mul_(conv_head_init_scale)
        self.conv_head.bias.data.mul_(conv_head_init_scale)

       ##### Global Branch Setting #######

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # The channels of stage4 output feature matrix
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(
                    patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                    norm_layer=nn.LayerNorm if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        # stochastic depth decay rule
        i_layer = 0 # layer index

        self.layers1 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint)
        

        i_layer = 1
        self.layers2 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 2
        self.layers3 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 3
        self.layers4 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        ###### Hierachical Feature Fusion Block Setting #######

        self.fu1 = Hifuse_block(input_channels=96, mid_channels=96, reduction_factor=16, intermediate_channels=96, output_channels=96, dropout_rate=HFF_dp)
        self.fu2 = Hifuse_block(input_channels=192, mid_channels=192, reduction_factor=16, intermediate_channels=192, output_channels=192, dropout_rate=HFF_dp)
        self.fu3 = Hifuse_block(input_channels=384, mid_channels=384, reduction_factor=16, intermediate_channels=384, output_channels=384, dropout_rate=HFF_dp)
        self.fu4 = Hifuse_block(input_channels=768, mid_channels=768, reduction_factor=16, intermediate_channels=768, output_channels=768, dropout_rate=HFF_dp)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        x_s, H, W = self.patch_embed(imgs)
        x_s = self.pos_drop(x_s)
        x_s_1, H, W = self.layers1(x_s, H, W)
        
        x_s_2, H, W = self.layers2(x_s_1, H, W)
        x_s_3, H, W = self.layers3(x_s_2, H, W)
        x_s_4, H, W = self.layers4(x_s_3, H, W)


        # [B,L,C] ---> [B,C,H,W]
        x_s_1 = torch.transpose(x_s_1, 1, 2)
        x_s_1 = x_s_1.view(x_s_1.shape[0], -1, 56, 56)
        
        x_s_2 = torch.transpose(x_s_2, 1, 2)
        x_s_2 = x_s_2.view(x_s_2.shape[0], -1, 28, 28)
        x_s_3 = torch.transpose(x_s_3, 1, 2)
        x_s_3 = x_s_3.view(x_s_3.shape[0], -1, 14, 14)
        x_s_4 = torch.transpose(x_s_4, 1, 2)
        x_s_4 = x_s_4.view(x_s_4.shape[0], -1, 7, 7)

        ######  Local Branch ######
        x_c = self.conv_layer_0(imgs)

        x_c_1 = self.local_block_1(x_c)
        x_c_1 = self.local_block_1(x_c_1)

        # Stage 2
        x_c = self.conv_layer_2(x_c_1)
        x_c_2 = self.local_block_2(x_c)
        x_c_2 = self.local_block_2(x_c_2)

        # Stage 3
        x_c = self.conv_layer_3(x_c_2)
        x_c_3 = self.local_block_3(x_c)
        x_c_3 = self.local_block_3(x_c_3)

        # Stage 4
        x_c = self.conv_layer_4(x_c_3)
        x_c_4 = self.local_block_4(x_c)        
        x_c_4 = self.local_block_4(x_c_4)        

        ###### Hierachical Feature Fusion Path ######
        x_f_1 = self.fu1(x_c_1, x_s_1, None)
        
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)

        x_fu = self.conv_norm(x_f_4.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        x_fu = self.conv_head(x_fu)

        return x_fu

def HiFuse_Small(num_classes: int):
    model = HifuseModel(depths=(2, 2, 6, 2),
                     conv_depths=(2, 2, 6, 2),
                     num_classes=num_classes)
    return model

def main():

    # Step stem processing
    image_path = 'hifuse-model-image-classifiaction/preprocessing/output_images/ISIC_0000064.jpg'
    # image_path = 'E:/research papers/coding/hifuse-model-image-classifiaction/preprocessing/0cbddd47-c7f6-474b-9e9a-b5d2429a2312.jpg'
    output_dir = './output_images'
    image_tensor = load_image(image_path)
    print("Image Tensor Shape:", image_tensor.shape)

    patch_extractor = Convolution_2d_k4_s4(in_channels=3, out_channels=96, kernel_size=4, stride=4)#
    with torch.no_grad():
        convolved_tensor = patch_extractor(image_tensor)
    # stem_layerNormalization = LayerNormChannelsFirst(96, eps=1e-6)#
    batch_size, num_channels, height, width = convolved_tensor.shape # (N, C, H, W)
    print(f"Dimensions of the convolved image: Batch Size={batch_size}, Channels={num_channels}, Height={height}, Width={width}")
    # convolved_tensor_local = stem_layerNormalization(convolved_tensor)
    # print(" After Layer Normalization:",convolved_tensor_local.shape)
    
    # Stage 1
    extractor = LocalFeatureExtractor(in_channels=96, out_channels=96, dropout_prob=0.0)
    local_branch_1 = extractor(convolved_tensor)
    print("After Local Feature Block",local_branch_1.shape)
    # Convert the tensor to separate channel images
    channel_images = tensor_to_channel_images(local_branch_1)
    # Visualize the channel images
    visualize_channel_images(channel_images)

    # Global Block

    i_layer = 0
    window_size=7
    embed_dim = 96
    depths=(2, 2, 2, 2)
    num_heads=(3, 6, 12, 24)
    qkv_bias=True
    drop_rate=0
    attn_drop_rate=0 
    drop_path_rate=0.
    norm_layer=nn.LayerNorm,
    patch_norm=True
    use_checkpoint=False
    patch_size=4
    in_chans=3
    embed_dim=96
    
    # stochastic depth
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    layers1 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer > 0) else None,
                                use_checkpoint=use_checkpoint)
    i_layer = 1
    layers2 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer > 0) else None,
                                use_checkpoint=use_checkpoint)

    i_layer = 2
    layers3 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer > 0) else None,
                                use_checkpoint=use_checkpoint)

    i_layer = 3
    layers4 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer > 0) else None,
                                use_checkpoint=use_checkpoint)

    patch_embed = PatchEmbed(
        patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
        norm_layer=nn.LayerNorm if patch_norm else None)
    pos_drop = nn.Dropout(p=drop_rate)

    x_s, H, W = patch_embed(image_tensor)
    x_s = pos_drop(x_s)
    global_branch_1, H, W = layers1(x_s, H, W)

    # [B,L,C] ---> [B,N,H,W]
    global_branch_1 = torch.transpose(global_branch_1, 1, 2)  # Transpose to [B, C, L]
    N = global_branch_1.shape[1]  # Number of channels
    # Assuming L = H * W, calculate H and W
    global_branch_1 = global_branch_1.reshape(global_branch_1.shape[0], N, H, W)  # Reshape to [B, N, H, W]
    print("After Global Feature Block",global_branch_1.shape)

    channel_images = tensor_to_channel_images(global_branch_1)
    # Visualize the channel images
    visualize_channel_images(channel_images)


    ###### Hierachical Feature Fusion Block Setting #######
    HFF_dp=0.
    fu1 = Hifuse_block(input_channels=96, mid_channels=96, reduction_factor=16, intermediate_channels=96, output_channels=96, dropout_rate=HFF_dp)

    ###### Hierachical Feature Fusion Path ######
    x_f_1 = fu1(local_branch_1, global_branch_1, None)
    print("After Hierarchical Feature Fusion Block",x_f_1.shape)
    channel_images = tensor_to_channel_images(x_f_1)
    # Visualize the channel images
    visualize_channel_images(channel_images)



if __name__ == '__main__':
    main()
