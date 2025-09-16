# --------------------------------------------------------
# Code adapted by Vanessa Borst, Timo Dittus and Contributors
# from https://github.com/yan-hao-tian/VW
#
# Original license from the VWFormer repository:
# The MIT License
# Copyright (c) 2024 yan-hao-tian
# --------------------------------------------------------

import torch
from torch.nn import functional as F

from medseg.evaluation.params import create_model_summary
from medseg.models.backbones.convnext import ConvNeXt
from medseg.models.decode_heads.vw_head import VWHead
from medseg.models.segmentors.segmentor import Segmentor

# Code adapted from https://github.com/yan-hao-tian/VW


class VWFormerConvNext(Segmentor):
    def __init__(
            self,
            in_size,
            in_channels=3,
            out_channels=1,
            conv_next_arch='small',
            vw_head_in_channels=[96, 192, 384, 768],
            vw_head_embed_dim=512,
            vw_head_n_heads=1,
            vw_head_short_cut=True,     # TODO: Check, by default True for ADE20K, False for Cityscapes
            pretrained_filename=None
    ):
        super().__init__()

        self.in_size = in_size
        self.conv_next = ConvNeXt(
            in_channels=in_channels,
            arch=conv_next_arch,
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            pretrained_filename=pretrained_filename
            # stem_patch_size=4,
            # norm_cfg=dict(type='LN2d', eps=1e-6),
            # act_cfg=dict(type='GELU'),
            # linear_pw_conv=True,
            # frozen_stages=0,
            # init_cfg=None
        )

        # SyncBN is BN for distributed training and used in the original implementation.
        # However, if only one GPU is used, using simple BN is faster.
        norm_cfg = dict(type='SyncBN' if torch.cuda.device_count() > 1 else 'BN', requires_grad=True)

        self.decoder = VWHead(
            in_channels=vw_head_in_channels,
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
        y = self.conv_next(x)
        y = self.decoder(y)
        y = F.interpolate(y, size=(self.in_size, self.in_size), mode='bilinear', align_corners=False)
        return y


class VWFormerConvNextS(VWFormerConvNext):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs,
                         conv_next_arch='small',
                         vw_head_in_channels=[96, 192, 384, 768],
                         vw_head_n_heads=16,
                         # "https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth"
                         pretrained_filename='convnext-small_in1k.pth')

class VWFormerConvNextB(VWFormerConvNext):
    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs,
                         conv_next_arch='base',
                         vw_head_in_channels=[128, 256, 512, 1024],
                         pretrained_filename='convnext-base_in1k.pth')



if __name__ == "__main__":
    model = VWFormerConvNextS(in_size=512, in_channels=3, out_channels=1)
    im = torch.randn(1, 3, 512, 512)
    print(create_model_summary(model, im.shape))
