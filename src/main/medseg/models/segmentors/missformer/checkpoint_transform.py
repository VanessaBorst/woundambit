import re
from collections import OrderedDict
import torch

from medseg.util.helper_functions import integer_float_to_int

# The following function is copied from
# https://github.com/open-mmlab/mmsegmentation/blob/main/tools/model_converters/mit2mmseg.py
# Copyright 2020 The MMSegmentation Authors. All rights reserved.
#                                  Apache License

# Aim: Convert SegFormer GitHub model weights to MMSegmentation style.
def convert_own_to_mmseg_mit(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        # patch embedding conversion
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i - 1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        # transformer encoder layer conversion
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            new_k = k.replace(f'block{stage_i}', f'layers.{stage_i - 1}.1')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                new_k = new_k.replace('fc2.', '4.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
        # norm layer conversion
        elif k.startswith('norm'):
            stage_i = int(k.split('.')[0].replace('norm', ''))
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i - 1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v
    return new_ckpt


# The following function was created to reverse the above changes (no official code, custom function)

def convert_mmseg_to_own_mit(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        # patch embedding conversion
        elif re.match(r"^layers\.\d+\.0", k):  # k.startswith('patch_embed'):
            stage_i = int(k.split('.')[1])
            new_k = k.replace(f'layers.{stage_i}.0', f'patch_embed{stage_i + 1}')
            # new_k = k.replace(f'patch_embed{stage_i}', f'layers.{stage_i - 1}.0')
            new_v = v
            if 'projection.' in new_k:
                new_k = new_k.replace('projection.', 'proj.')
        # transformer encoder layer conversion
        elif re.match(r"^layers\.\d+\.1", k):  # k.startswith('block'):
            stage_i = int(k.split('.')[1])
            new_k = k.replace(f'layers.{stage_i}.1', f'block{stage_i + 1}')
            new_v = v
            if 'attn.in_proj_' in new_k:  # 'attn.q.' in new_k:
                # Revert the stacking from above (torch.cat([query, kv], dim=0))

                d1 = integer_float_to_int(v.shape[0] / 3)  # Get original size of q

                if new_v.dim() == 2:
                    # Weight
                    q_recovered = new_v[:d1, :]  # First d1 rows
                    kv_recovered = new_v[d1:, :]  # Remaining rows
                elif new_v.dim() == 1:
                    # Bias
                    q_recovered = new_v[:d1]
                    kv_recovered = new_v[d1:]
                else:
                    raise ValueError("Could not convert tensor to new shape")

                # Write the value for both directly and continue
                new_ckpt[new_k.replace('attn.in_proj_', 'q.')] = q_recovered
                new_ckpt[new_k.replace('attn.in_proj_', 'kv.')] = kv_recovered
                continue
            elif 'attn.out_proj.' in new_k:
                new_k = new_k.replace('attn.out_proj.', 'proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('attn.sr.', 'attn.sr.')
            elif 'ffn.layers.' in new_k:
                new_k = new_k.replace('ffn.layers.', 'mlp.')

                if '0.weight' in new_k or '4.weight' in new_k:
                    new_v = new_v.squeeze(-1).squeeze(-1)
                    # new_v = v.reshape((*v.shape, 1, 1))

                new_k = new_k.replace('mlp.1.', 'mlp.dwconv.dwconv.')
                new_k = new_k.replace('mlp.0.', 'mlp.fc1.')
                new_k = new_k.replace('mlp.4.', 'mlp.fc2.')
        # norm layer conversion
        elif re.match(r"^layers\.\d+\.2", k):  # k.startswith('norm'):
            stage_i = int(k.split('.')[1])
            new_k = k.replace(f'layers.{stage_i}.2', f'norm{stage_i + 1}', )
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v
    return new_ckpt
