# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# NOTE: InternImage and UPerNet have dedicated licenses, see respective classes.
# --------------------------------------------------------

import torch
from torch import nn

from medseg.evaluation.params import create_model_summary
from medseg.models.common.resize import resize
from medseg.models.backbones.internimage import (
    InternImageT, InternImageS, InternImageB, InternImageL, InternImageXL, InternImageH
)
from medseg.models.decode_heads.uper_head import UPerHead
from medseg.models.segmentors.segmentor import Segmentor

# Mapping for decoder in-channels
DECODER_IN_CHANNELS = {
    "tiny": [64, 128, 256, 512],
    "small": [80, 160, 320, 640],
    "base": [112, 224, 448, 896],
    "large": [160, 320, 640, 1280],
    "extra_large": [192, 384, 768, 1536],
    "huge": [320, 640, 1280, 2560]
}

# Mapping for encoder classes
ENCODER_CLASSES = {
    "tiny": InternImageT,
    "small": InternImageS,
    "base": InternImageB,
    "large": InternImageL,
    "extra_large": InternImageXL,
    "huge": InternImageH
}


class InternImageUperNet(Segmentor):
    def __init__(self, in_size=512, intern_image_encoder="tiny", in_channels=3,
                 out_channels=1, decoder_in_channels=None):
        super().__init__()
        self.in_size = in_size

        # If string, initialize encoder
        if isinstance(intern_image_encoder, str):
            intern_image_encoder = intern_image_encoder.lower()
            if intern_image_encoder not in ENCODER_CLASSES:
                raise ValueError(f"Unknown intern image encoder: {intern_image_encoder}")
            self.intern_image = ENCODER_CLASSES[intern_image_encoder](in_channels=in_channels)
        elif isinstance(intern_image_encoder, nn.Module):
            self.intern_image = intern_image_encoder
        else:
            raise TypeError(f"Invalid type for intern_image_encoder: {type(intern_image_encoder)}")

        # SyncBN is BN for distributed training and used in the original implementation.
        # However, if only one GPU is used, using simple BN is faster.
        norm_cfg = dict(type='SyncBN' if torch.cuda.device_count() > 1 else 'BN', requires_grad=True)

        # Default decoder channels if not provided
        self.decoder = UPerHead(
            in_channels=decoder_in_channels or DECODER_IN_CHANNELS[intern_image_encoder],
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
        if not isinstance(self.intern_image, nn.Module):
            raise TypeError(f"Expected nn.Module for intern_image, got {type(self.intern_image)}")
        y = self.intern_image(x)
        seg_logits = self.decoder(y)
        y = resize(input=seg_logits,
                   size=self.in_size,
                   mode='bilinear',
                   align_corners=False)
        return y


# Subclasses
class InternImageUperNet_T(InternImageUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, intern_image_encoder="tiny")


class InternImageUperNet_S(InternImageUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, intern_image_encoder="small")


class InternImageUperNet_B(InternImageUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, intern_image_encoder="base")


class InternImageUperNet_L(InternImageUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, intern_image_encoder="large")


class InternImageUperNet_XL(InternImageUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, intern_image_encoder="extra_large")


class InternImageUperNet_H(InternImageUperNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, intern_image_encoder="huge")


if __name__ == "__main__":
    device = torch.device('cuda:0')

    model = InternImageUperNet_T(in_size=512, in_channels=3, out_channels=1)
    model.to(device)

    im = torch.randn(1, 3, 512, 512).to(device)
    print(create_model_summary(model, im.shape, device=device, depth=2))
    # y = model(im)
    # print(y.shape)
    # print(get_total_params(model))
