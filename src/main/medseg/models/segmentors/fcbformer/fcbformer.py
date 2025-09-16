# --------------------------------------------------------
# Code adapted by Vanessa Borst, Timo Dittus and Contributors
# from https://github.com/ESandML/FCBFormer/
#
# Original license from the FCBFormer repository:
# Apache License 2.0
# Authors: Edward Sanderson and Bogdan J. Matuszewski
# --------------------------------------------------------


from functools import partial
from typing import Callable

import torch
from timm.models.vision_transformer import _cfg
from torch import nn, Tensor
from math import log

from medseg.evaluation.params import create_model_summary
from medseg.models.backbones.pvt_v2 import PyramidVisionTransformerV2
from medseg.models.segmentors.segmentor import Segmentor
from medseg.training.loss.soft_dice_loss import SoftDiceLoss
from medseg.util.path_builder import get_root_path


# Adapted from  https://github.com/ESandML/FCBFormer
class FCBFormer(Segmentor):
    """
    FCBFormer with PVTv2-b3 backbone. From paper: FCN-Transformer Feature Fusion for Polyp Segmentation (
    https://doi.org/10.1007/978-3-031-12053-4_65). Authors originally used a resolution of 352x352, batch size of 16,
    AdamW optimizer, learning rate of 1e-4, 200 epochs and a scheduler that halved the learning rate if validation
    mDice did not improve over 10 epochs until a lr minimum of 1e-6 was reached. Code adapted from
    https://github.com/ESandML/FCBFormer
    """

    def __init__(self,
                 in_size=352,
                 in_channels=3,
                 out_channels=1,
                 fcb_min_level_channels=32,
                 fcb_n_levels=6,
                 fcb_n_residual_block_per_level=2,
                 ph_rb_channels=(64, 64),
                 use_eca=False):
        super().__init__()

        # calculate fcb channel mults dynamically
        fcb_min_channel_mults = []
        current_mult = 1
        while len(fcb_min_channel_mults) < fcb_n_levels:
            fcb_min_channel_mults.extend([current_mult] * 2)
            current_mult *= 2
        fcb_min_channel_mults = tuple(fcb_min_channel_mults[:fcb_n_levels])

        self.TB = TransformerBranch(use_eca=use_eca)

        self.FCB = FullyConvolutionalBranch(
            in_channels=in_channels,
            in_size=in_size,
            min_level_channels=fcb_min_level_channels,
            min_channel_mults=fcb_min_channel_mults,
            n_levels_down=fcb_n_levels,
            n_levels_up=fcb_n_levels,
            n_residual_block_per_level=fcb_n_residual_block_per_level,
            use_eca=use_eca
        )

        self.PH = PredictionHead(
            rb_channels=ph_rb_channels,
            use_eca=use_eca,
            out_channels=out_channels
        )

        self.up_tosize = nn.Upsample(size=in_size)

    def default_loss_func(self, multiclass: bool) -> Callable[[Tensor, Tensor], Tensor]:
        if not multiclass:
            def loss_func_binary(predictions: Tensor, targets: Tensor) -> Tensor:
                return SoftDiceLoss()(predictions, targets) + torch.nn.BCEWithLogitsLoss()(predictions, targets)

            return loss_func_binary

        else:
            def loss_func_multiclass(predictions: Tensor, targets: Tensor) -> Tensor:
                return SoftDiceLoss()(predictions, targets) + torch.nn.CrossEntropyLoss()(
                    torch.softmax(predictions, dim=1), targets)

            return loss_func_multiclass

    def forward(self, x):
        x1 = self.TB(x)  # x1 shape = (batch_size, 64, 88, 88)
        x2 = self.FCB(x)  # x2 shape = (batch_size, 32, in_size, in_size)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out


class PredictionHead(nn.Module):
    def __init__(self,
                 in_channels=96,
                 out_channels=1,
                 rb_channels=(64, 64),
                 use_eca=False):
        super().__init__()
        self.rb_channels = []

        if isinstance(rb_channels, int):
            self.rb_channels.append(rb_channels)
        elif isinstance(rb_channels, tuple):
            self.rb_channels.extend(rb_channels)

        n_rb_blocks = len(self.rb_channels)

        self.res_blocks = nn.Sequential()
        for i in range(n_rb_blocks):
            rb_in_ch = in_channels if i == 0 else self.rb_channels[i - 1]
            rb_out_ch = self.rb_channels[i]
            self.res_blocks.append(ResidualBlock(rb_in_ch, rb_out_ch, use_eca=use_eca))

        out_conv_in_ch = in_channels if n_rb_blocks == 0 else self.rb_channels[-1]
        self.out_conv = nn.Conv2d(out_conv_in_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.out_conv(x)
        return x


# FCB: Fully Convolutional Branch
class FullyConvolutionalBranch(nn.Module):
    def __init__(
            self,
            in_size,
            in_channels=3,
            min_level_channels=32,
            min_channel_mults=(1, 1, 2, 2, 4, 4),
            n_levels_down=6,
            n_levels_up=6,
            n_residual_block_per_level=2,
            use_eca=False,
    ):

        super().__init__()
        # Resolution gets halved (n_levels_down -1) times, because of the Down-Convolutions.
        # So we have to check if the resolution is a multiple of 2^(n_levels_down -1), which is 32 per default.
        assertion_multiple = 2 ** (n_levels_down - 1)
        assert in_size % assertion_multiple == 0, f"Resolution has to be a multiple of {assertion_multiple}."

        self.enc_blocks = nn.ModuleList([nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)])

        ch = min_level_channels
        enc_block_chans = [min_level_channels]

        # Add Residual Blocks for each level in the down path
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            # loop from 0 to 1
            for block in range(n_residual_block_per_level):
                self.enc_blocks.append(
                    ResidualBlock(ch, min_channel_mult * min_level_channels, use_eca=use_eca)
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)

            # Add Down-Convolutions for all levels except the last one
            if level != n_levels_down - 1:
                self.enc_blocks.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))  # halves resolution
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(ResidualBlock(ch, ch, use_eca=use_eca),
                                          ResidualBlock(ch, ch, use_eca=use_eca))

        self.dec_blocks = nn.ModuleList([])

        # Add Residual Blocks for each level in the up path
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_residual_block_per_level + 1):
                layers = [
                    ResidualBlock(ch + enc_block_chans.pop(), min_channel_mult * min_level_channels, use_eca=use_eca)]
                ch = min_channel_mult * min_level_channels

                # Add Up-Convolutions for all levels except the last one, add them only after the last Residual Block
                if level < n_levels_up - 1 and block == n_residual_block_per_level:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for i, module in enumerate(self.enc_blocks):
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class ECA(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        self.t = int(abs((log(in_channels, 2) + b)))
        self.k = self.t if self.t % 2 else self.t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k, padding=int(self.k / 2), bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


# RB: Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_eca: bool = False):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ECA(out_channels) if use_eca else nn.Identity(),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class TransformerBranch(nn.Module):
    def __init__(self,
                 embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4],
                 depths=[3, 4, 18, 3],
                 use_eca=False):
        super().__init__()

        # Instantiate the PyramidVisionTransformerV2 backbone with the given parameters
        backbone = PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=depths,
            sr_ratios=[8, 4, 2, 1],
        )

        # Load the pre-trained model checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(f"{get_root_path()}/data/pretrained/pvt_v2_b3.pth", map_location=device)
        backbone.default_cfg = _cfg()  # TODO: not sure if this is needed
        backbone.load_state_dict(checkpoint)

        # Remove the last layer of the pre-trained model and create a new sequential model
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        # Replace ModuleList layers 1, 4, 7, 10 with Sequential layers (keeping the children the same)
        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        # Define the Local Emphasis (LE) layers
        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    ResidualBlock([64, 128, 320, 512][i], 64, use_eca=use_eca),
                    ResidualBlock(64, 64, use_eca=use_eca),
                    nn.Upsample(size=88)
                )
            )

        # Define the Stepwise Feature Aggregation (SFA) layers
        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(
                ResidualBlock(128, 64, use_eca=use_eca),
                ResidualBlock(64, 64, use_eca=use_eca)))

    def get_pyramid(self, x):
        # Extracts the feature pyramid from the input tensor
        pyramid = []
        B = x.shape[0]  # Batch size
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:  # Input shape: (B, C, H, W); Output shape: (B, H, W, C)
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:  # Input shape: (B, H, W, C); Output shape: (B, H, W, C)
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:  # Input shape: (B, H, W, C); Output shape: (B, C, H, W)
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        # Pass the input through the feature pyramid extraction
        pyramid = self.get_pyramid(x)
        pyramid_emph = []

        # Apply Local Emphasis (LE) layers to the feature pyramid
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

            # Initialize the final output with the last level of the enhanced pyramid
        l_i = pyramid_emph[-1]

        # Iteratively apply Stepwise Feature Aggregation (SFA) layers to aggregate features
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i),
                          dim=1)  # Concatenate the current level with the aggregated features
            l = self.SFA[i](l)  # Apply the SFA layer
            l_i = l  # Update the aggregated features

        return l  # Return the final output tensor with aggregated features


if __name__ == "__main__":
    model = FCBFormer(in_size=512, in_channels=3, out_channels=1)  # , num_heads=[2,4,10,16])
    im = torch.randn(1, 3, 512, 512)
    print(create_model_summary(model, im.shape, depth=2))
