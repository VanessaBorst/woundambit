import torch
from torch import nn


# Adapted from https://github.com/YuWenLo/HarDNet-DFUS (kingnet.py)
# VB: Most important changes:
# ConvLayer: Removed unused dropout param; nn.Sequential -> nn.Module (but keep layers identical!)
# Renamed KingBlock -> HarDBlockV2
# Moved KingNet and KingNet53 to hardnet_dfus.py and renamed KingNet53 to HarDNet53


def channel_split(x, split):
    return torch.split(x, split, dim=1)


# %%
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0), -1)


# %%
class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


# %%
class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = in_channels
        kernel = 3
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                            stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)

    # %%


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.conv = nn.Conv2d(in_channels, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2,
                              groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.relu(x)

    # %%


class SELayer(nn.Module):
    def __init__(self, channel, reduction=20):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # squeeze
        self.fc = nn.Sequential(  # excitation
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # scale
        '''
        return x'''


# %%
class CSPKingBlock(nn.Module):
    def get_out_ch(self):
        return self.kingouch + self.patial_channel

    def __init__(self, in_channels, n_layers, patial_channel, dwconv=False):
        super().__init__()
        self.patial_channel = patial_channel
        self.KingBlk = HarDBlockV2(in_channels - patial_channel, n_layers, dwconv=dwconv)
        ouch = self.KingBlk.get_out_ch()
        self.kingouch = ouch
        self.transition = nn.Sequential(SELayer(ouch),
                                        ConvLayer(ouch, ouch, kernel=1))

    def forward(self, x):
        x1 = x[:, :self.patial_channel, :, :]  # cross stage
        x2 = x[:, self.patial_channel:, :, :]
        x2 = self.KingBlk(x2)
        x2 = self.transition(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


# %%
class HarDBlockV2(nn.Module):
    def get_divisor(self):  # get the divisors of n
        divisors = []
        for i in range(1, self.n_layers + 1):
            if (int(self.n_layers / i) == self.n_layers / i):
                divisors.append(i)
        return divisors

    def get_link(self):  # calculate the linkcount of all layer
        links = [[] for x in range(self.n_layers)]
        for div in self.divisors:
            for k in range(0, self.n_layers, div):
                links[k].append(div)
        return links

    def get_out_ch(self):  # calculate the outchannel of block
        link_count = 0
        for out in self.concate_out[self.n_layers]:
            link_count += len(self.links[out])
        return self.growth * link_count + self.in_channels

    def __init__(self, in_channels, n_layers, dwconv=False):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.divisors = self.get_divisor()
        self.links = self.get_link()
        self.concate_out = {3: [2], 4: [2], 6: [2, 4], 8: [2, 6], 9: [3, 6], 10: [2, 5, 8], 12: [3, 6, 9],
                            15: [3, 6, 9, 12], 16: [2, 6, 10, 14]}
        self.growth = int(self.in_channels / len(self.divisors))
        layers_ = []
        for i in range(n_layers):
            if (i != n_layers - 1):
                channel = len(self.links[i + 1]) * self.growth
            else:
                channel = self.in_channels
            if dwconv:
                layers_.append(CombConvLayer(channel, channel))
            else:
                layers_.append(ConvLayer(channel, channel))
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        tensors = [[] for i in range(self.n_layers)]
        for layer in range(len(self.layers)):
            # tins = torch.split(x, self.growth, dim=1)  # VB: Changed to channel split, should be the same
            tins = channel_split(x, self.growth)
            for i in range(len(tins)):
                tensors[layer + self.links[layer][i] - 1].append(tins[i])
            if len(tensors[layer]) > 1:
                x = torch.cat(tensors[layer], dim=1)
            else:
                x = tensors[layer][0]
            x = self.layers[layer](x)

            layers_.append(x)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == t - 1) or (i in self.concate_out[self.n_layers]):
                out_.append(layers_[i])

        out = torch.cat(out_, 1)
        # ----------不用改----------
        return out
