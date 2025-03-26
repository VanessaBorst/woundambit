import torch

from medseg.evaluation.params import create_model_summary
from medseg.models.segmentors.segmentor import Segmentor
from medseg.models.decode_heads.ham_decoder import LightHamHead
from medseg.models.backbones.mscan_encoder import MSCAN


# Code adapted from https://github.com/Visual-Attention-Network/SegNeXt

class SegNeXt(Segmentor):
    def __init__(self,
                 in_size,
                 in_channels=3,
                 out_channels=1,
                 mscan_embed_dims=[64, 128, 320, 512],
                 mscan_mlp_ratios=[8, 8, 4, 4],
                 mscan_dropout_ratio=0.0,
                 mscan_drop_path_rate=0.1,
                 mscan_depths=[3, 3, 5, 2],
                 ham_in_channels=[64, 160, 256],
                 ham_in_index=[1, 2, 3],
                 ham_align_channels=256,  # renamed from channels to align channels
                 ham_channels=256,
                 ham_dropout_ratio=0.1,
                 ham_norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 ham_align_corners=False,
                 ham_kwargs=dict(),
                 pretrained_filename='mscan_b.pth',
                 ):
        super().__init__()
        self.mscan_encoder = MSCAN(in_chans=in_channels, embed_dims=mscan_embed_dims, mlp_ratios=mscan_mlp_ratios,
                                   drop_rate=mscan_dropout_ratio, drop_path_rate=mscan_drop_path_rate,
                                   depths=mscan_depths,
                                   pretrained_filename=pretrained_filename)

        self.ham_decoder = LightHamHead(out_size=in_size,
                                        in_channels=ham_in_channels,
                                        in_index=ham_in_index,
                                        ham_channels=ham_channels,
                                        channels=ham_align_channels,
                                        dropout_ratio=ham_dropout_ratio,
                                        align_corners=ham_align_corners,
                                        norm_cfg=ham_norm_cfg,
                                        out_channels=out_channels,
                                        num_classes=out_channels if out_channels != 1 else 2,  # Binary Seg.
                                        ham_kwargs=ham_kwargs)

    def forward(self, x):
        y = self.mscan_encoder(x)
        y = self.ham_decoder(y)
        return y


class SegNeXtL(SegNeXt):
    def __init__(self, *args, **kwargs, ):
        ham_channels = kwargs.pop('ham_channels', 1024)
        ham_align_channels = kwargs.pop('ham_align_channels', 1024)
        super().__init__(*args, **kwargs,
                         mscan_depths=[3, 5, 27, 3],
                         mscan_drop_path_rate=0.3,
                         ham_in_channels=[128, 320, 512],
                         ham_align_channels=ham_channels,
                         ham_channels=ham_align_channels,
                         pretrained_filename='mscan_l.pth')


class SegNeXtB(SegNeXt):
    def __init__(self, *args, **kwargs):
        ham_channels = kwargs.pop('ham_channels', 512)
        ham_align_channels = kwargs.pop('ham_align_channels', 512)
        super().__init__(*args, **kwargs,
                         mscan_depths=[3, 3, 12, 3],
                         ham_in_channels=[128, 320, 512],
                         ham_align_channels=ham_channels,
                         ham_channels=ham_align_channels,
                         pretrained_filename='mscan_b.pth')


class SegNeXtS(SegNeXt):
    def __init__(self, *args, **kwargs, ):
        ham_channels = kwargs.pop('ham_channels', 256)
        ham_align_channels = kwargs.pop('ham_align_channels', 256)
        super().__init__(*args, **kwargs,
                         mscan_depths=[2, 2, 4, 2],             # VB: Fixed from [2, 2, 6, 2] (18.02.2025)
                         ham_in_channels=[128, 320, 512],       # VB: Fixed from [64, 160, 256] (18.02.2025)
                         ham_align_channels=ham_channels,
                         ham_channels=ham_align_channels,
                         ham_kwargs=dict(MD_R=16),
                         pretrained_filename='mscan_s.pth')


class SegNeXtT(SegNeXt):
    def __init__(self, *args, **kwargs, ):
        ham_channels = kwargs.pop('ham_channels', 256)
        ham_align_channels = kwargs.pop('ham_align_channels', 256)
        super().__init__(*args, **kwargs,
                         mscan_depths=[3, 3, 5, 2],
                         mscan_embed_dims=[32, 64, 160, 256],           # VB: Added 18.02.2025
                         ham_in_channels=[64, 160, 256],
                         ham_align_channels=ham_channels,
                         ham_channels=ham_align_channels,
                         ham_kwargs=dict(MD_R=16),
                         pretrained_filename='mscan_t.pth')


if __name__ == "__main__":
    model = SegNeXtL(in_size=512, in_channels=3, out_channels=1)
    im = torch.randn(1, 3, 512, 512)
    print(create_model_summary(model, im.shape, device='cuda'))
