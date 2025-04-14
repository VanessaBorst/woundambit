import os
from typing import List, Callable

import torch
import torch.nn as nn
import wget
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from medseg.evaluation.params import create_model_summary
from medseg.models.segmentors.hiformer.hiformer_dec import ConvUpsample, HiFormerSegmentationHead
from medseg.models.segmentors.hiformer.hiformer_enc import All2Cross
from medseg.models.segmentors.segmentor import Segmentor
from medseg.training.loss.hiformer_dice_loss import HiFormerDiceLoss
from medseg.util.path_builder import PathBuilder

# Code adapted from https://github.com/amirhossein-kz/HiFormer

# The following is the same for all three model size variants (L, B, S)
SWIN_TINY_WEIGHTS_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0" \
                        "/swin_tiny_patch4_window7_224.pth"

import torch.nn.functional as F


class HiFormer(Segmentor):
    def __init__(self,
                 in_size: int = 224,
                 in_channels=3,
                 out_channels=1,
                 swin_pretrained_filename: str = "swin_tiny_patch4_window7_224.pth",
                 swin_pyramid_fm: List = [96, 192, 384],
                 use_resnet_pretrained: bool = True,
                 resnet_type: str = "resnet50",
                 cnn_pyramid_fm: List = [256, 512, 1024],
                 patch_size: int = 4,
                 depths=[[1, 2, 0]],
                 num_heads=(6, 12),
                 mlp_ratio=(2., 2., 1.),
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 cross_pos_embed=True,
                 # dec_channels = [16]
                 ):
        super().__init__()
        pretrained_pb = PathBuilder.pretrained_dir_builder()
        swin_pretrained_path = pretrained_pb.clone().add(swin_pretrained_filename).build()
        if not os.path.isfile(swin_pretrained_path):
            print(f"Swin-Tiny weights not found under {swin_pretrained_path}")
            print('Trying to download...')
            wget.download(SWIN_TINY_WEIGHTS_URL, swin_pretrained_path)
        self.in_size = in_size
        self.out_channels = out_channels
        self.pred_size = 224  # TODO VB: Check, can this be adapted to work with in_size?
        self.patch_size = [4, 16]

        self.All2Cross = All2Cross(
            swin_pretrained_path=swin_pretrained_path,
            use_resnet_pretrained=use_resnet_pretrained,
            resnet_type=resnet_type,
            in_size=self.pred_size,
            in_channels=in_channels,
            swin_pyramid_fm=swin_pyramid_fm,
            patch_size=patch_size,
            cnn_pyramid_fm=cnn_pyramid_fm,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            cross_pos_embed=cross_pos_embed)

        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)

        self.segmentation_head = HiFormerSegmentationHead(
            # dec_channels=dec_channels,
            in_channels=16,
            out_channels=self.out_channels,
            kernel_size=3,
        )

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def default_loss_func(self, multiclass: bool) -> Callable[[Tensor, Tensor], Tensor]:
        if not multiclass:
            bcel = BCEWithLogitsLoss()
            dice_loss = HiFormerDiceLoss(self.out_channels)

            def hiformer_loss(outputs, label_batch):
                loss_bcel = bcel(outputs, label_batch)
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_bcel + 0.6 * loss_dice
                return loss

            return hiformer_loss

        else:
            ce_loss = CrossEntropyLoss()
            dice_loss = HiFormerDiceLoss(self.out_channels)

            def hiformer_loss(outputs, label_batch):
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice
                return loss

            return hiformer_loss

    def forward(self, x):
        if self.pred_size != self.in_size:
            x = F.interpolate(x, size=(self.pred_size, self.pred_size), mode='bilinear', align_corners=False)
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            embed = Rearrange('b (h w) d -> b d h w', h=(self.pred_size // self.patch_size[i]),
                              w=(self.pred_size // self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)

            reshaped_embed.append(embed)

        C = reshaped_embed[0] + reshaped_embed[1]
        C = self.conv_pred(C)

        out = self.segmentation_head(C)
        if self.pred_size != self.in_size:
            out = F.interpolate(out, size=(self.in_size, self.in_size), mode='bilinear', align_corners=False)
        return out


class HiFormerL(HiFormer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         swin_pyramid_fm=[96, 192, 384],
                         cnn_pyramid_fm=[64, 128, 256],
                         depths=[[1, 4, 0]],
                         num_heads=(6, 6),
                         mlp_ratio=(2., 2., 1.),
                         resnet_type="resnet34"
                         )


class HiFormerB(HiFormer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         swin_pyramid_fm=[96, 192, 384],
                         cnn_pyramid_fm=[256, 512, 1024],
                         depths=[[1, 2, 0]],
                         num_heads=(6, 12),
                         mlp_ratio=(2., 2., 1.),
                         resnet_type="resnet50",
                         )


class HiFormerS(HiFormer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         swin_pyramid_fm=[96, 192, 384],
                         cnn_pyramid_fm=[64, 128, 256],
                         depths=[[1, 1, 0]],
                         num_heads=(3, 3),
                         mlp_ratio=(1., 1., 1.),
                         resnet_type="resnet34",
                         )


if __name__ == "__main__":
    model = HiFormerB(in_size=512, in_channels=3, out_channels=1)  # B4:  63,993,793 params
    im = torch.randn(1, 3, 512, 512)
    print(create_model_summary(model, im.shape))
