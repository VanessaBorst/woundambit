import torch
from torch import nn
from torchvision.models import VGG13_Weights

from medseg.evaluation.params import create_model_summary
from medseg.models.segmentors.segmentor import Segmentor
from medseg.util.path_builder import PathBuilder


class UNet(Segmentor):

    def __init__(self,
                 in_size,
                 in_channels=3,
                 out_channels=1,
                 l1_feature_maps=64,
                 unet_levels=4,
                 norm='none',
                 activation='relu',
                 use_residual=False,
                 use_pretrained=False):

        super(UNet, self).__init__()
        self.use_residual = use_residual
        self.in_size = in_size
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels
        assert in_size % 2 ** unet_levels == 0, \
            f"Image size {in_size} has to be divisible by 2^(unet_levels - 1) = 2^{unet_levels}"
        self.conv_block = ResidualBlock if use_residual else BasicConvBlock
        act_str = activation.lower()
        if act_str == 'relu':
            self.activation = nn.ReLU
        elif act_str == 'leakyrelu':
            self.activation = nn.LeakyReLU
        elif act_str == 'gelu':
            self.activation = nn.GELU
        elif act_str == 'silu':
            self.activation = nn.SiLU
        else:
            raise ValueError(f"Activation {activation} could not be mapped to an activation function")

        feature_cfg = []

        for i in range(unet_levels):
            # double the feature maps at each level
            feature_cfg.append(l1_feature_maps * 2 ** i)

        # Down blocks
        down_in_channels = self.in_channels
        for feature in feature_cfg:
            self.downs.append(self.conv_block(down_in_channels, feature, norm=norm, activation=self.activation))
            down_in_channels = feature

        # Bottleneck a.k.a bridge
        self.bottleneck = self.conv_block(feature_cfg[-1], feature_cfg[-1] * 2, norm=norm, activation=self.activation)

        # Up blocks
        for feature in reversed(feature_cfg):
            self.ups.append(
                UpBlock(feature * 2, feature, norm=norm, activation=self.activation, default_block=self.conv_block)
            )

        # Final convolution
        self.final_conv = nn.Conv2d(feature_cfg[0], out_channels, kernel_size=1)

        if use_pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        weights_mapping = {
            'downs.0.conv1.weight': 'features.0.weight',
            'downs.0.conv1.bias': 'features.0.bias',
            'downs.0.conv2.weight': 'features.2.weight',
            'downs.0.conv2.bias': 'features.2.bias',
            'downs.1.conv1.weight': 'features.5.weight',
            'downs.1.conv1.bias': 'features.5.bias',
            'downs.1.conv2.weight': 'features.7.weight',
            'downs.1.conv2.bias': 'features.7.bias',
            'downs.2.conv1.weight': 'features.10.weight',
            'downs.2.conv1.bias': 'features.10.bias',
            'downs.2.conv2.weight': 'features.12.weight',
            'downs.2.conv2.bias': 'features.12.bias',
            'downs.3.conv1.weight': 'features.15.weight',
            'downs.3.conv1.bias': 'features.15.bias',
            'downs.3.conv2.weight': 'features.17.weight',
            'downs.3.conv2.bias': 'features.17.bias'
        }
        vgg13 = torch.hub.load('pytorch/vision', 'vgg13', weights=VGG13_Weights.IMAGENET1K_V1)
        state_dict_to_load = dict()
        for k, v in weights_mapping.items():
            if k in self.state_dict():
                state_dict_to_load[k] = vgg13.state_dict()[v]
                # print(f"Mapped {k} to {v}")
        self.load_state_dict(state_dict_to_load, strict=False)
        del vgg13

    def forward(self, x):
        skip_connections = []
        y = x
        for i in range(0, len(self.downs)):
            y = self.downs[i](y)
            skip_connections.append(y)
            y = self.pool(y)

        y = self.bottleneck(y)

        # reverse list
        skip_connections = list(reversed(skip_connections))

        # loop through up-layers in steps of 2
        for i in range(0, len(self.ups)):
            y = self.ups[i](y, skip_connections[i])
        return self.final_conv(y)


class Norm(nn.Module):
    def __init__(self, norm='none', channels=None, groups=None):
        super().__init__()
        if norm == 'batch':
            assert channels is not None
            self.norm = nn.BatchNorm2d(channels)
        elif norm == 'group':
            assert channels is not None and groups is not None
            if channels % groups != 0:
                self.norm = nn.Identity()
            else:
                self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(x)


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='none', activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = Norm(norm=norm, channels=out_channels, groups=32)
        self.activation = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = Norm(norm=norm, channels=out_channels, groups=32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1x1=True, use_preactivation_pattern=True, strides=1,
                 norm='none',
                 activation=nn.ReLU):
        super().__init__()
        self.use_preactivation_pattern = use_preactivation_pattern
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides) if use_conv1x1 else None
        self.norm1 = Norm(norm=norm, channels=in_channels if use_preactivation_pattern else out_channels, groups=32)
        self.norm2 = Norm(norm=norm, channels=out_channels, groups=32)
        self.activation = activation()

    def forward(self, x):
        if self.use_preactivation_pattern:
            y = self.norm1(x)
            y = self.activation(y)
            y = self.conv1(y)
            y = self.norm2(y)
            y = self.activation(y)
            y = self.conv2(y)
            concat = self.conv3(x) if self.conv3 else x
            y += concat
        else:
            y = self.conv1(x)
            y = self.norm1(y)
            y = self.activation(y)
            y = self.conv2(y)
            y = self.norm2(y)
            concat = self.conv3(x) if self.conv3 else x
            y += concat
            y = self.activation(y)
        return y


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='none', activation=nn.ReLU,
                 default_block: type = BasicConvBlock):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = default_block(in_channels, out_channels, norm=norm, activation=activation)

    def forward(self, x, x_concat):
        y = self.up_conv(x)
        y = torch.cat((x_concat, y), dim=1)
        y = self.conv_block(y)
        return y


if __name__ == "__main__":
    im = torch.randn(5, 3, 224, 224)
    model = UNet(in_size=224)
    print(create_model_summary(model, im.shape))
