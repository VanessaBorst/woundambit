# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from medseg.models.common.resize import resize


# Code adapted from https://github.com/NVlabs/SegFormer
# Changed: Replaced dropout and last layer linear_pred with conv_seg from BaseModule (as it is usual)
# Added embed dim param directly (instead of decoder_params with only one entry)

from medseg.models.decode_heads.decode_head import BaseDecodeHead


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, mlp_embed_dim, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.mlp_embed_dim = mlp_embed_dim

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.mlp_embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.mlp_embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.mlp_embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.mlp_embed_dim)

        # SyncBN is BN for distributed training and used in the original implementation.
        # However, if only one GPU is used, using simple BN is faster.
        norm_cfg = dict(type='SyncBN' if torch.cuda.device_count() > 1 else 'BN', requires_grad=True)
        self.linear_fuse = ConvModule(
            in_channels=self.mlp_embed_dim * 4,
            out_channels=self.mlp_embed_dim,
            kernel_size=1,
            norm_cfg=norm_cfg
        )

        # self.linear_pred = nn.Conv2d(self.mlp_embed_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        # input is list of tensors, multilevel features
        # inputs transformed with multiple_select, which simply means that only list items with indices in self.in_index are selected
        # in_index from config is [0, 1, 2, 3]
        x = self._transform_inputs(x)  # len=4, 1/4,1/8,1/16,1/32

        # split into c1, c2, c3, c4
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # put through MLP layer then swap from 0,1,2 to 0,2,1 then reshape to n, ... , something
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # next, resize it to height and width of c1
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.conv_seg(_c)        # Replaces the following two calls
        # x = self.dropout(_c)
        # x = self.linear_pred(x)

        return x
