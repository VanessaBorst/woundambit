# --------------------------------------------------------
# Code adapted by Vanessa Borst, Timo Dittus and Contributors
# from  https://github.com/NVlabs/SegFormer
#
# Original license from the SegFormer repository:
# NVIDIA Source Code License
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# --------------------------------------------------------

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from medseg.evaluation.params import create_model_summary
from medseg.models.backbones.mit import MixVisionTransformer
from medseg.models.decode_heads.segformer_decoder import SegFormerHead
from medseg.models.segmentors.segmentor import Segmentor


# Code adapted from https://github.com/NVlabs/SegFormer and default params for each variant copied from configs

class Segformer(Segmentor):
    def __init__(
            self,
            in_size,
            in_channels=3,
            out_channels=1,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            encoder_drop_rate=0.0,
            decoder_drop_rate=0.1,
            encoder_drop_path_rate=0.1,
            mlp_embed_dim=768,
            pretrained_filename=None
    ):
        super().__init__()

        self.in_size = in_size
        self.mit = MixVisionTransformer(
            img_size=in_size,
            in_chans=in_channels,
            num_classes=out_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            drop_rate=encoder_drop_rate,
            drop_path_rate=encoder_drop_path_rate,
            norm_layer=norm_layer,
            depths=depths,
            sr_ratios=sr_ratios,
            pretrained_filename=pretrained_filename
        )

        # SyncBN is BN for distributed training and used in the original implementation.
        # However, if only one GPU is used, using simple BN is faster.
        norm_cfg = dict(type='SyncBN' if torch.cuda.device_count() > 1 else 'BN', requires_grad=True)

        self.decoder = SegFormerHead(
            in_channels=embed_dims,
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=mlp_embed_dim,     # Warning: In Segformer configs always set to 128, seems to be a bug
            dropout_ratio=decoder_drop_rate,  # 0.1,
            out_channels=out_channels,
            num_classes=out_channels if out_channels != 1 else 2,  # Binary Seg.
            norm_cfg=norm_cfg,
            align_corners=False,
            mlp_embed_dim=mlp_embed_dim  # 768
        )

    def forward(self, x):
        y = self.mit(x)
        y = self.decoder(y)
        y = F.interpolate(y, size=(self.in_size, self.in_size), mode='bilinear', antialias=True, align_corners=False)
        return y


class SegformerB0(Segformer):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2], mlp_embed_dim=256,
                         pretrained_filename='mit_b0.pth')


class SegformerB1(Segformer):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, depths=[2, 2, 2, 2], mlp_embed_dim=256,pretrained_filename='mit_b1.pth')


class SegformerB2(Segformer):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, depths=[3, 4, 6, 3], pretrained_filename='mit_b2.pth')


class SegformerB3(Segformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depths=[3, 4, 18, 3], pretrained_filename='mit_b3.pth')


class SegformerB4(Segformer):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, depths=[3, 8, 27, 3], pretrained_filename='mit_b4.pth')


class SegformerB5(Segformer):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, depths=[3, 6, 40, 3], pretrained_filename='mit_b5.pth')


if __name__ == "__main__":
    model = SegformerB3(in_size=512, in_channels=3, out_channels=1)  # B4:  63,993,793 params
    im = torch.randn(1, 3, 512, 512)
    print(create_model_summary(model, im.shape))
