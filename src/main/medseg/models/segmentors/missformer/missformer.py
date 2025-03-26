import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from medseg.evaluation.params import create_model_summary
from medseg.models.segmentors.missformer.missformer_components import TransformerBlock, M_EfficientSelfAtten, \
    MixFFN_skip, MiT
from medseg.models.segmentors.segmentor import Segmentor
# Adapted from https://github.com/ZhifangDeng/MISSFormer/
# Modified to make it also work with 512-sized images (not only 224-sized images)
# Instead of hard-coded intervals, the intervals are calculated based on the input size (im_size and channel dims)
# Included pretraining (original implementation trains from scratch)
# Removed class SegU_decoder, BridgeLayer_3 and BridegeBlock_3 (not used)
# Introduced in_channels as parameter
from medseg.util.helper_functions import integer_float_to_int


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class BridgeLayer_4(nn.Module):
    def __init__(self, im_size, dims, head, reduction_ratios, channel_factors, spatial_dims, intervals):
        super().__init__()
        self.im_size = im_size
        self.channel_factors = channel_factors
        self.spatial_dims = spatial_dims
        self.intervals = intervals

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios, channel_factors, spatial_dims)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims, dims * 4)
        self.mixffn2 = MixFFN_skip(dims * channel_factors[0], dims * channel_factors[0] * 4)
        self.mixffn3 = MixFFN_skip(dims * channel_factors[1], dims * channel_factors[1] * 4)
        self.mixffn4 = MixFFN_skip(dims * channel_factors[2], dims * channel_factors[2] * 4)

    def forward(self, inputs):
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            # (im_size / 4) ^ 2,
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64 (for 224, else 16384*64 for 512)
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64 (for 224, else 8192*64 for 512)
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64 (for 224, else 5120*64 for 512)
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64 (for 224, else 2048*64 for 512)

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem1 = tx[:, :self.intervals[0], :].reshape(B, -1, C)
        tem2 = tx[:, self.intervals[0]:self.intervals[1], :].reshape(B, -1, C * self.channel_factors[0])
        tem3 = tx[:, self.intervals[1]:self.intervals[2], :].reshape(B, -1, C * self.channel_factors[1])
        tem4 = tx[:, self.intervals[2]:self.intervals[3], :].reshape(B, -1, C * self.channel_factors[2])

        m1f = self.mixffn1(tem1, self.spatial_dims[0], self.spatial_dims[0]).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, self.spatial_dims[1], self.spatial_dims[1]).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, self.spatial_dims[2], self.spatial_dims[2]).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, self.spatial_dims[3], self.spatial_dims[3]).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)

        tx2 = tx1 + t1

        return tx2


class BridegeBlock_4(nn.Module):
    def __init__(self, im_size, dims, head, reduction_ratios, channel_factors):
        super().__init__()
        self.im_size = im_size
        self.channel_factors = channel_factors

        # For 224: 56, 28, 14, 7
        # For 512: 128, 64, 32, 16
        self.spatial_dims = [integer_float_to_int((self.im_size / 4)),
                             integer_float_to_int((self.im_size / 8)),
                             integer_float_to_int((self.im_size / 16)),
                             integer_float_to_int((self.im_size / 32))]

        interval_1_end = self.spatial_dims[0] ** 2  # 3136, 16384 (values are for dims=[64, 128, 320, 512])
        interval_2_end = interval_1_end + self.channel_factors[0] * self.spatial_dims[1] ** 2  # 4704, 24576
        interval_3_end = interval_2_end + self.channel_factors[1] * self.spatial_dims[2] ** 2  # 5684, 29696
        interval_4_end = interval_3_end + self.channel_factors[2] * self.spatial_dims[3] ** 2  # 6076, 31744
        self.intervals = [interval_1_end, interval_2_end, interval_3_end, interval_4_end]

        self.bridge_layer1 = BridgeLayer_4(im_size, dims, head, reduction_ratios, channel_factors, self.spatial_dims,
                                           self.intervals)
        self.bridge_layer2 = BridgeLayer_4(im_size, dims, head, reduction_ratios, channel_factors, self.spatial_dims,
                                           self.intervals)
        self.bridge_layer3 = BridgeLayer_4(im_size, dims, head, reduction_ratios, channel_factors, self.spatial_dims,
                                           self.intervals)
        self.bridge_layer4 = BridgeLayer_4(im_size, dims, head, reduction_ratios, channel_factors, self.spatial_dims,
                                           self.intervals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape
        outs = []

        sk1 = bridge4[:, :self.intervals[0], :].reshape(B, self.spatial_dims[0], self.spatial_dims[0], C).permute(0, 3,
                                                                                                                  1, 2)
        sk2 = bridge4[:, self.intervals[0]:self.intervals[1], :].reshape(B, self.spatial_dims[1], self.spatial_dims[1],
                                                                         C * self.channel_factors[0]).permute(0, 3, 1,
                                                                                                              2)
        sk3 = bridge4[:, self.intervals[1]:self.intervals[2], :].reshape(B, self.spatial_dims[2], self.spatial_dims[2],
                                                                         C * self.channel_factors[1]).permute(0, 3, 1,
                                                                                                              2)
        sk4 = bridge4[:, self.intervals[2]:self.intervals[3], :].reshape(B, self.spatial_dims[3], self.spatial_dims[3],
                                                                         C * self.channel_factors[2]).permute(0, 3, 1,
                                                                                                              2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
            # self.last_layer = None

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            # print("------",x1.shape, x2.shape)
            cat_x = torch.cat([x1, x2], dim=-1)
            # print("-----catx shape", cat_x.shape)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out


class MISSFormer(Segmentor):
    def __init__(self, in_size=512,
                 in_channels=3,
                 out_channels=1,  # num_classes
                 token_mlp_mode="mix_skip",
                 encoder_pretrained=False,
                 # If set to True, 512 images will be downscaled
                 operate_on_224=False):
        # use_pretrained=False):
        super().__init__()

        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        assert in_size in [224, 512], "MISSFormer is proposed for input size 224 or 512"

        self.downsampled_to_224 = False
        if in_size == 512 and operate_on_224:
            # Internal operation size will be 224 x 224
            self.downsampled_to_224 = True
            in_size = 224

        d_base_feat_size = 7  if in_size == 224 else 16  # 16 for 512 input size; 7 for 224

        # Warning: Dim does not match the description in the paper (should be 64, 128, 256, 512)
        # See issue https://github.com/ZhifangDeng/MISSFormer/issues/11
        # However, [64, 128, 256, 512] does not work in combination with heads=[1, 2, 5, 8] for MiT
        # Error: dim 256 should be divided by num_heads 5.
        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]

        # The following values are NOT part of the original implementation
        # heads = [1, 2, 4, 8]
        # dims, layers = [[64, 128, 256, 512], [2, 2, 2, 2]]
        # in_out_chan = [[32, 64], [128, 128], [256, 256], [512, 512]]

        self.backbone = MiT(image_size=in_size,
                            in_channels=in_channels,
                            dims=dims,
                            layers=layers,
                            token_mlp=token_mlp_mode,
                            heads=heads,
                            # If true, use ImageNet weights for the encoder
                            encoder_pretrained=encoder_pretrained)

        self.reduction_ratios = [1, 2, 4, 8]
        channel_factors_from_base_channels = [int(dims[1] / dims[0]), int(dims[2] / dims[0]), int(dims[3] / dims[0])]
        self.bridge = BridegeBlock_4(in_size, 64, 1, self.reduction_ratios,
                                     channel_factors=channel_factors_from_base_channels)

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], heads[3],
                                        reduction_ratios[3], token_mlp_mode, n_class=out_channels)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2], heads[2],
                                        reduction_ratios[2], token_mlp_mode, n_class=out_channels)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], heads[1],
                                        reduction_ratios[1], token_mlp_mode, n_class=out_channels)
        self.decoder_0 = MyDecoderLayer((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], heads[0],
                                        reduction_ratios[0], token_mlp_mode, n_class=out_channels, is_last=True)

        # if use_pretrained:
        #     checkpoint_path = PathBuilder.pretrained_dir_builder().add("missformer_isic2018.pt").build()
        #     checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        #
        #     for k in list(checkpoint.keys()):
        #         if k not in self.state_dict().keys() or "decoder_0.last_layer." in k:
        #             del checkpoint[k]
        #
        #     # The weights of the last layer are not contained
        #     self.load_state_dict(checkpoint, strict=False)

    def forward(self, x):

        if self.downsampled_to_224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', antialias=True, align_corners=False)

        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder = self.backbone(x)
        bridge = self.bridge(encoder)  # list

        b, c, _, _ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        # ---------------Decoder-------------------------
        # print("stage3-----")
        tmp_3 = self.decoder_3(bridge[3].permute(0, 2, 3, 1).view(b, -1, c))
        # print("stage2-----")
        tmp_2 = self.decoder_2(tmp_3, bridge[2].permute(0, 2, 3, 1))
        # print("stage1-----")
        tmp_1 = self.decoder_1(tmp_2, bridge[1].permute(0, 2, 3, 1))
        # print("stage0-----")
        tmp_0 = self.decoder_0(tmp_1, bridge[0].permute(0, 2, 3, 1))

        if self.downsampled_to_224:
            tmp_0 = F.interpolate(tmp_0, size=(512, 512), mode='bilinear', antialias=True, align_corners=False)

        return tmp_0


if __name__ == "__main__":
    SIZE = 512
    im = torch.randn(1, 3, SIZE, SIZE)
    model = MISSFormer(in_size=SIZE, in_channels=3, out_channels=1, encoder_pretrained=True)
    print(create_model_summary(model, im.shape, depth=1))
