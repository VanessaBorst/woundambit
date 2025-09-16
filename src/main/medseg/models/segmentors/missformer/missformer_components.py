# --------------------------------------------------------
# Code (possibly) adapted by Vanessa Borst, Timo Dittus and Contributors
# from https://github.com/ZhifangDeng/MISSFormer/
#
# Original license from the MISSFormer repository: Not specified
# --------------------------------------------------------

import torch
from torch import nn

from medseg.models.segmentors.missformer.checkpoint_transform import convert_mmseg_to_own_mit
from medseg.util.helper_functions import integer_float_to_int
from medseg.util.path_builder import PathBuilder


class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        assert C % self.head == 0, f"Channels {C} must be divisible by heads {self.head}"
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


# class SelfAtten(nn.Module):
#     def __init__(self, dim, head):
#         super().__init__()
#         self.head = head
#         self.scale = (dim // head) ** -0.5
#         self.q = nn.Linear(dim, dim, bias=True)
#         self.kv = nn.Linear(dim, dim * 2, bias=True)
#         self.proj = nn.Linear(dim, dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
#
#         kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn_score = attn.softmax(dim=-1)
#
#         x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj(x_atten)
#
#         return out


class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio, channel_factors, spatial_dims):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        self.channel_factors = channel_factors
        self.spatial_dims = spatial_dims

        if (len(self.reduction_ratio) == 4):
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * channel_factors[0], dim * channel_factors[0], reduction_ratio[2],
                                 reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * channel_factors[1], dim * channel_factors[1], reduction_ratio[1],
                                 reduction_ratio[1])

        elif (len(self.reduction_ratio) == 3):
            self.sr0 = nn.Conv2d(dim * channel_factors[0], dim * channel_factors[0], reduction_ratio[2],
                                 reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * channel_factors[1], dim * channel_factors[1], reduction_ratio[1],
                                 reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        if (len(self.reduction_ratio) == 4):
            interval_1_end = self.spatial_dims[0] ** 2  # 3136, 16384 (values are for dims=[64, 128, 320, 512])
            interval_2_end = interval_1_end + self.channel_factors[0] * self.spatial_dims[1] ** 2  # 4704, 24576
            interval_3_end = interval_2_end + self.channel_factors[1] * self.spatial_dims[2] ** 2  # 5684, 29696
            interval_4_end = interval_3_end + self.channel_factors[2] * self.spatial_dims[3] ** 2  # 6076, 31744

            tem0 = x[:, :interval_1_end, :].reshape(B, self.spatial_dims[0], self.spatial_dims[0], C).permute(0, 3, 1,
                                                                                                              2)
            tem1 = x[:, interval_1_end:interval_2_end, :].reshape(B, self.spatial_dims[1], self.spatial_dims[1],
                                                                  C * self.channel_factors[0]).permute(0, 3, 1, 2)
            tem2 = x[:, interval_2_end:interval_3_end, :].reshape(B, self.spatial_dims[2], self.spatial_dims[2],
                                                                  C * self.channel_factors[1]).permute(0, 3, 1, 2)
            tem3 = x[:, interval_3_end:interval_4_end, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if (len(self.reduction_ratio) == 3):
            # TODO: Check when using it in the future
            interval_1_end = integer_float_to_int((self.spatial_dims[0] ** 2) / 2)  # 1568, 8192
            interval_2_end = interval_1_end + self.channel_factors[1] * self.spatial_dims[2] ** 2  # 2548, 13312
            interval_3_end = interval_2_end + self.channel_factors[2] * self.spatial_dims[3] ** 2  # 2940, 15360

            tem0 = x[:, :interval_1_end, :].reshape(B, self.spatial_dims[1], self.spatial_dims[1],
                                                    C * self.channel_factors[0]).permute(0, 3, 1, 2)
            tem1 = x[:, interval_1_end:interval_2_end, :].reshape(B, self.spatial_dims[2], self.spatial_dims[2],
                                                                  C * self.channel_factors[1]).permute(0, 3, 1, 2)
            tem2 = x[:, interval_2_end:interval_3_end, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio, channel_factors, spatial_dims):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio, channel_factors, spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


#
# class LocalEnhance_EfficientSelfAtten(nn.Module):
#     def __init__(self, dim, head, reduction_ratio):
#         super().__init__()
#         self.head = head
#         self.reduction_ratio = reduction_ratio
#         self.scale = (dim // head) ** -0.5
#         self.q = nn.Linear(dim, dim, bias=True)
#         self.kv = nn.Linear(dim, dim * 2, bias=True)
#         self.proj = nn.Linear(dim, dim)
#         self.local_pos = DWConv(dim)
#
#         if reduction_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
#
#         if self.reduction_ratio > 1:
#             p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
#             sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
#             x = self.norm(sp_x)
#
#         kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn_score = attn.softmax(dim=-1)
#         local_v = v.permute(0, 2, 1, 3).reshape(B, N, C)
#         local_pos = self.local_pos(local_v).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)
#         x_atten = ((attn_score @ v) + local_pos).transpose(1, 2).reshape(B, N, C)
#         out = self.proj(x_atten)
#
#         return out


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# class MixD_FFN(nn.Module):
#     def __init__(self, c1, c2, fuse_mode="add"):
#         super().__init__()
#         self.fc1 = nn.Linear(c1, c2)
#         self.dwconv = DWConv(c2)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(c2, c1) if fuse_mode == "add" else nn.Linear(c2 * 2, c1)
#         self.fuse_mode = fuse_mode
#
#     def forward(self, x):
#         ax = self.dwconv(self.fc1(x), H, W)
#         fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == "add" else self.act(torch.cat([ax, self.fc1(x)], 2))
#         out = self.fc2(ax)
#         return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


# class FuseTransformerBlock(nn.Module):
#     def __init__(self, dim, head, reduction_ratio=1, fuse_mode="add"):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = MixD_FFN(dim, int(dim * 4), fuse_mode)
#
#     def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
#         tx = x + self.attn(self.norm1(x), H, W)
#         mx = tx + self.mlp(self.norm2(tx), H, W)
#         return mx


# class MLP(nn.Module):
#     def __init__(self, dim, embed_dim):
#         super().__init__()
#         self.proj = nn.Linear(dim, embed_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.flatten(2).transpose(1, 2)
#         return self.proj(x)


# class ConvModule(nn.Module):
#     def __init__(self, c1, c2, k):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.activate = nn.ReLU(True)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.activate(self.bn(self.conv(x)))


# def _revert_mit(ckpt):
#     new_ckpt = OrderedDict()
#
#     # Process the concat between q linear weights and kv linear weights
#     for k, v in ckpt.items():
#         if k.startswith('head'):
#             continue
#         # patch embedding conversion
#         elif k.startswith('layers'):
#             stage_i = int(k.split('.')[0].replace('layers', '')) + 1
#             new_k = k.replace(f'layers.{stage_i - 1}.0', f'patch_embed{stage_i}')
#             new_v = v
#             if 'projection.' in new_k:
#                 new_k = new_k.replace('projection.', 'proj.')
#         # transformer encoder layer conversion
#         elif k.startswith('layers'):
#             stage_i = int(k.split('.')[0].replace('layers', '')) + 1
#             new_k = k.replace(f'layers.{stage_i - 1}.1', f'block{stage_i}')
#             new_v = v
#             if 'attn.in_proj_' in new_k:
#                 sub_item_k = k.replace('attn.in_proj_', 'attn.q.')
#                 new_k = new_k.replace('attn.in_proj_', 'attn.q.')
#                 new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
#             elif 'attn.out_proj.' in new_k:
#                 new_k = new_k.replace('attn.out_proj.', 'attn.proj.')
#             elif 'sr.' in new_k:
#                 new_k = new_k.replace('sr.', 'attn.sr.')
#             elif 'ffn.layers.' in new_k:
#                 string = f'{new_k}-'
#                 new_k = new_k.replace('ffn.layers.', 'mlp.')
#                 if '0.' in new_k or '4.' in new_k:
#                     new_v = v.reshape((*v.shape, 1, 1))
#                 new_k = new_k.replace('0.', 'fc1.')
#                 new_k = new_k.replace('1.', 'dwconv.dwconv.')
#                 new_k = new_k.replace('4.', 'fc2.')
#                 string += f'{new_k} {v.shape}-{new_v.shape}'
#         # norm layer conversion
#         elif k.startswith('layers'):
#             stage_i = int(k.split('.')[0].replace('layers', '')) + 1
#             new_k = k.replace(f'layers.{stage_i - 1}.2', f'norm{stage_i}')
#             new_v = v
#         else:
#             new_k = k
#             new_v = v
#         new_ckpt[new_k] = new_v
#     return new_ckpt


class MiT(nn.Module):
    def __init__(self, image_size, in_channels, dims, layers, token_mlp='mix_skip', heads=[1, 2, 5, 8],
                 encoder_pretrained=False):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        # heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0],
                                                   in_channels, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
                                                   dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
                                                   dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
                                                   dims[2], dims[3])

        # transformer encoder
        self.block1 = nn.ModuleList([
            TransformerBlock(dims[0], heads[0], reduction_ratios[0], token_mlp)
            for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(dims[1], heads[1], reduction_ratios[1], token_mlp)
            for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp)
            for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

        # Warning: Pretraining is not used in the original implementation, but added here for a fair benchmarking
        if encoder_pretrained:
            checkpoint_path = PathBuilder.pretrained_dir_builder().add("mit_b1.pth").build()
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available()
                                                                               else 'cpu'))

            checkpoint = convert_mmseg_to_own_mit(checkpoint)
            # own_weights = self.state_dict()

            # checkpoint_renamed = _revert_mit(checkpoint)
            # remove keys that do not belong to the model
            for k in list(checkpoint.keys()):
                if k not in self.state_dict().keys():
                    del checkpoint[k]
            # The loaded state dict has 48 keys less (since the MixFFN_skip has 3 LayerNorms in addition)
            # NumLayers per FFN * 2 (weight/bias) * 2 (2 per block) * 4 (num Blocks)
            # 3 * 2 * 2 * 4 = 48
            self.load_state_dict(checkpoint, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

#
# class FuseMiT(nn.Module):
#     def __init__(self, image_size, dims, layers, fuse_mode='add'):
#         super().__init__()
#         patch_sizes = [7, 3, 3, 3]
#         strides = [4, 2, 2, 2]
#         padding_sizes = [3, 1, 1, 1]
#         reduction_ratios = [8, 4, 2, 1]
#         heads = [1, 2, 5, 8]
#
#         # patch_embed
#         self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
#         self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
#                                                    dims[0], dims[1])
#         self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
#                                                    dims[1], dims[2])
#         self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
#                                                    dims[2], dims[3])
#
#         # transformer encoder
#         self.block1 = nn.ModuleList([
#             FuseTransformerBlock(dims[0], heads[0], reduction_ratios[0], fuse_mode)
#             for _ in range(layers[0])])
#         self.norm1 = nn.LayerNorm(dims[0])
#
#         self.block2 = nn.ModuleList([
#             FuseTransformerBlock(dims[1], heads[1], reduction_ratios[1], fuse_mode)
#             for _ in range(layers[1])])
#         self.norm2 = nn.LayerNorm(dims[1])
#
#         self.block3 = nn.ModuleList([
#             FuseTransformerBlock(dims[2], heads[2], reduction_ratios[2], fuse_mode)
#             for _ in range(layers[2])])
#         self.norm3 = nn.LayerNorm(dims[2])
#
#         self.block4 = nn.ModuleList([
#             FuseTransformerBlock(dims[3], heads[3], reduction_ratios[3], fuse_mode)
#             for _ in range(layers[3])])
#         self.norm4 = nn.LayerNorm(dims[3])
#
#         # self.head = nn.Linear(dims[3], num_classes)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B = x.shape[0]
#         outs = []
#
#         # stage 1
#         x, H, W = self.patch_embed1(x)
#         for blk in self.block1:
#             x = blk(x, H, W)
#         x = self.norm1(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)
#
#         # stage 2
#         x, H, W = self.patch_embed2(x)
#         for blk in self.block2:
#             x = blk(x, H, W)
#         x = self.norm2(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)
#
#         # stage 3
#         x, H, W = self.patch_embed3(x)
#         for blk in self.block3:
#             x = blk(x, H, W)
#         x = self.norm3(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)
#
#         # stage 4
#         x, H, W = self.patch_embed4(x)
#         for blk in self.block4:
#             x = blk(x, H, W)
#         x = self.norm4(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)
#
#         return outs


# class Decoder(nn.Module):
#     def __init__(self, dims, embed_dim, num_classes):
#         super().__init__()
#
#         self.linear_c1 = MLP(dims[0], embed_dim)
#         self.linear_c2 = MLP(dims[1], embed_dim)
#         self.linear_c3 = MLP(dims[2], embed_dim)
#         self.linear_c4 = MLP(dims[3], embed_dim)
#
#         self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, 1)
#         self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
#
#         self.conv_seg = nn.Conv2d(128, num_classes, 1)
#
#         self.dropout = nn.Dropout2d(0.1)
#
#     def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         c1, c2, c3, c4 = inputs
#         n = c1.shape[0]
#         c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
#
#         c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         c2f = F.interpolate(c2f, size=c1.shape[2:], mode='bilinear', align_corners=False)
#
#         c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         c3f = F.interpolate(c3f, size=c1.shape[2:], mode='bilinear', align_corners=False)
#
#         c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         c4f = F.interpolate(c4f, size=c1.shape[2:], mode='bilinear', align_corners=False)
#
#         c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
#         c = self.dropout(c)
#         return self.linear_pred(c)


# segformer_settings = {
#     'B0': [[32, 64, 160, 256], [2, 2, 2, 2], 256],  # [channel dimensions, num encoder layers, embed dim]
#     'B1': [[64, 128, 320, 512], [2, 2, 2, 2], 256],
#     'B2': [[64, 128, 320, 512], [3, 4, 6, 3], 768],
#     'B3': [[64, 128, 320, 512], [3, 4, 18, 3], 768],
#     'B4': [[64, 128, 320, 512], [3, 8, 27, 3], 768],
#     'B5': [[64, 128, 320, 512], [3, 6, 40, 3], 768]
# }


# class SegFormer(nn.Module):
#     def __init__(self, model_name: str = 'B0', num_classes: int = 19, image_size: int = 224) -> None:
#         super().__init__()
#         assert model_name in segformer_settings.keys(), f"SegFormer model name should be in {list(segformer_settings.keys())}"
#         dims, layers, embed_dim = segformer_settings[model_name]
#
#         self.backbone = MiT(image_size, dims, layers)
#         self.decode_head = Decoder(dims, embed_dim, num_classes)
#
#     def init_weights(self, pretrained: str = None) -> None:
#         if pretrained:
#             self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
#         else:
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.LayerNorm):
#                     nn.init.ones_(m.weight)
#                     nn.init.zeros_(m.bias)
#                 elif isinstance(m, nn.Conv2d):
#                     nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.size()[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#         encoder_outs = self.backbone(x)
#         return self.decode_head(encoder_outs)
