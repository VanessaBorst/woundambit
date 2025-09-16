# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# NOTE: TransNeXt and UPerNet have dedicated licenses, see respective classes.
# --------------------------------------------------------


import torch
from torch import nn

from medseg.evaluation.params import create_model_summary
from medseg.models.backbones.transnext import TransNextModel
from medseg.models.common.resize import resize

from medseg.models.decode_heads.uper_head import UPerHead
from medseg.models.segmentors.segmentor import Segmentor

# Mapping for decoder in-channels
DECODER_IN_CHANNELS = {
    "tiny": [72, 144, 288, 576],
    "small": [72, 144, 288, 576],
    "base": [96, 192, 384, 768]
}


# Notice about the is_extrapolation param:
# From the original repo regarding ADE20K: "Our TransNeXt models are trained with single-scale images,
# if you want to reproduce accurately, please use <config-file-ending-with-ss>."
# -> We stick to this config file, setting is_extrapolation to False (and img_size to 512)

class TransNextUperNet(Segmentor):
    def __init__(self, in_size=512, transnext_encoder="tiny",
                 in_channels=3, transnext_is_extrapolation=False,
                 out_channels=1, decoder_in_channels=None):
        super().__init__()
        self.in_size = in_size

        # If string, initialize encoder
        if isinstance(transnext_encoder, str):
            transnext_encoder = transnext_encoder.lower()
            if transnext_encoder not in DECODER_IN_CHANNELS.keys():
                raise ValueError(f"Unknown intern image encoder: {transnext_encoder}")
            self.transnext = TransNextModel(variant=transnext_encoder,
                                            in_channels=in_channels,
                                            img_size=in_size,
                                            is_extrapolation=transnext_is_extrapolation)
        else:
            raise TypeError(f"Invalid type for intern_image_encoder: {type(transnext_encoder)}")

        # SyncBN is BN for distributed training and used in the original implementation.
        # However, if only one GPU is used, using simple BN is faster.
        norm_cfg = dict(type='SyncBN' if torch.cuda.device_count() > 1 else 'BN', requires_grad=True)

        # Default decoder channels if not provided
        self.decoder = UPerHead(
            in_channels=decoder_in_channels or DECODER_IN_CHANNELS[transnext_encoder],
            # The following params are same for all variants
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            out_channels=out_channels,
            num_classes=out_channels if out_channels != 1 else 2,  # Binary Seg.
            norm_cfg=norm_cfg,
            align_corners=False
        )

    def forward(self, x):
        if not isinstance(self.transnext, nn.Module):
            raise TypeError(f"Expected nn.Module for TransNeXt, got {type(self.transnext)}")
        y = self.transnext(x)
        seg_logits = self.decoder(y)
        y = resize(input=seg_logits,
                   size=self.in_size,
                   mode='bilinear',
                   align_corners=False)
        return y


# Subclasses
class TransNeXtUperNet_Tiny(TransNextUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, transnext_encoder="tiny")


class TransNeXtUperNet_Small(TransNextUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, transnext_encoder="small")


class TransNeXtUperNet_Base(TransNextUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, transnext_encoder="base")


if __name__ == "__main__":
    device = torch.device('cuda:0')  # torch.device('cpu')
    model = TransNeXtUperNet_Tiny(in_size=512, in_channels=3, out_channels=1)
    model.to(device)
    im = torch.randn(1, 3, 512, 512).to(device)
    print(create_model_summary(model, im.shape, device=device, depth=2))
