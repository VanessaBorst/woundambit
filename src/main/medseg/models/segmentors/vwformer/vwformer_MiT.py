from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from medseg.evaluation.params import create_model_summary
from medseg.models.backbones.mit import MixVisionTransformer
from medseg.models.segmentors.segmentor import Segmentor

# Code adapted from https://github.com/yan-hao-tian/VW
from medseg.models.decode_heads.vw_head import VWHead


class VWFormerMiT(Segmentor):
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
            encoder_drop_path_rate=0.1,
            vw_head_embed_dim=512,
            vw_head_n_heads=1,       # TODO: Check - Varies in detailed configs (default: 1)
            vw_head_short_cut=True,  # TODO: Check: Varies (False for Cityscape,True for ADE20K but not in all configs)
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

        self.decoder = VWHead(
            in_channels=embed_dims,  # TODO: Check; should be [64, 128, 320, 512] for MiT-B3?

            in_index=[0, 1, 2, 3],
            channels=vw_head_embed_dim,
            dropout_ratio=0.1,
            out_channels=out_channels,
            num_classes=out_channels if out_channels != 1 else 2,  # Binary Seg.

            short_cut=vw_head_short_cut,
            nheads=vw_head_n_heads,

            norm_cfg=norm_cfg,
            align_corners=False,
        )

    def forward(self, x):
        y = self.mit(x)
        y = self.decoder(y)
        y = F.interpolate(y, size=(self.in_size, self.in_size), mode='bilinear', align_corners=False)
        return y


class VWFormerMiTB0(VWFormerMiT):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs,
                         embed_dims=[32, 64, 160, 256],
                         depths=[2, 2, 2, 2],
                         vw_head_embed_dim=128,  # See appendix of Paper
                         pretrained_filename='mit_b0.pth')


class VWFormerMiTB1(VWFormerMiT):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs,
                         depths=[2, 2, 2, 2],
                         vw_head_embed_dim=128,  # See appendix of Paper
                         pretrained_filename='mit_b1.pth')


class VWFormerMiTB2(VWFormerMiT):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, depths=[3, 4, 6, 3], pretrained_filename='mit_b2.pth')


class VWFormerMiTB3(VWFormerMiT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depths=[3, 4, 18, 3], pretrained_filename='mit_b3.pth')


class VWFormerMiTB4(VWFormerMiT):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, depths=[3, 8, 27, 3], pretrained_filename='mit_b4.pth')


class VWFormerMiTB5(VWFormerMiT):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs, depths=[3, 6, 40, 3], pretrained_filename='mit_b5.pth')


if __name__ == "__main__":
    model = VWFormerMiTB3(in_size=512, in_channels=3, out_channels=1)
    im = torch.randn(1, 3, 512, 512)
    print(create_model_summary(model, im.shape))
