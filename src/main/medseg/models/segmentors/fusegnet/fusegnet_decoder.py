import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


# Since the authors integrated custom attention type into the decoder that are not part of the official repo, we copied
# there modified versions here

# ==============================================================================
# Modified SCSE module supporing different strategies
class SCSEModule0(nn.Module):
    def __init__(self, in_channels, reduction=16, strategy='addition'):
        super().__init__()
        self.strategy = strategy
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            # nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        self.if_concat = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):

        xc = x * self.cSE(x)  # cSE attention
        xs = x * self.sSE(x)  # sSE attention

        if self.strategy == 'addition':
            x = xc + xs

        elif self.strategy == 'maxout':
            x = torch.maximum(xc, xs)

        elif self.strategy == 'concat':
            x = torch.cat([xc, xs], dim=1)
            x = self.if_concat(x)

        elif self.strategy == 'multiplication':
            x = xc * xs

        elif self.strategy == 'average':
            x = torch.stack((xc, xs), dim=0)
            x = torch.mean(x, dim=0)

        elif self.strategy == 'all-average':
            x1 = xc + xs  # addition
            x2 = torch.maximum(xc, xs)  # maxout
            x3 = self.if_concat(torch.cat([xc, xs], dim=1))  # concatenation
            x4 = xc * xs  # multiplication

            # Take average of above mentioned strategies
            x = torch.stack((x1, x2, x3, x4), dim=0)
            x = torch.mean(x, dim=0)

        else:
            raise ValueError(
                "Wrong keyword for attention strategy. Choose from [addition, maxout, concat, multiplication, average, all-average]")

        return x


# ==============================================================================
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()

        self.attention_type = attention_type

        self.conv1 = md.Conv2dReLU(
            # 2 * (in_channels + skip_channels), # -------->> use for concatenation <<----------
            in_channels + skip_channels,  # -------->> use for addition <<----------
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # P-scSE
        if self.attention_type == 'pscse':
            self.attention1 = SCSEModule0(in_channels=in_channels + skip_channels, strategy='maxout')
            self.attention2 = SCSEModule0(in_channels=in_channels + skip_channels, strategy='addition')

        # scSE
        elif self.attention_type == 'scse':
            self.attention1 = md.Attention("scse", in_channels=in_channels + skip_channels)
            self.attention2 = md.Attention("scse", in_channels=out_channels)

        # Single attention (as per the original paper -> attention is applied at the end)
        elif self.attention_type in ('maxout', 'additive', 'concat', 'multiplication', 'average', 'all-average'):
            self.attention = SCSEModule0(in_channels=out_channels, strategy=self.attention_type)

        else:
            raise ValueError("Wrong keyword for attention. Choose either of - pscse, scse, \
                             maxout, additive, concat, multiplication, average, all-average")

    def forward(self, x, skip=None, strategy='add'):

        # For P-scSE
        if self.attention_type == 'pscse':
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
                x1 = self.attention1(x)  # 1st attention
                x2 = self.attention2(x)  # 2nd attention

                if strategy == 'cat':
                    x3 = torch.cat([x1, x2], dim=1)  # concatenate two attentions
                elif strategy == 'add':
                    x3 = x1 + x2

            else:
                x2 = self.attention2(x)  # 2nd attention
                if strategy == 'cat':
                    x3 = torch.cat([x, x2], dim=1)  # concatenate x and x2
                elif strategy == 'add':
                    x3 = x + x2

            x3 = self.conv1(x3)

            return x3

        # For scSE
        elif self.attention_type == 'scse':
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
                x = self.attention1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.attention2(x)
            return x

        # Single attention (as per the original paper -> attention is applied at the end)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.attention(x)
            return x


# =========================================================================
class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
