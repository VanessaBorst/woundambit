# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from copy import deepcopy
from typing import Tuple, Optional, Union

import torch
from beartype import beartype
from torch import nn
from torchinfo import torchinfo

from medseg.util.files import save_text_to_file


def get_model_args(cfg: dict):
    model_args = deepcopy(cfg['architecture'])
    if 'arch_type' in model_args: del model_args['arch_type']
    if 'model_name' in model_args: del model_args['model_name']
    return model_args


def get_total_params(model: nn.Module) -> int:
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    return total_params


@beartype
def create_model_summary(model: nn.Module, input_size: Tuple, device: Union[str, torch.device] = 'cpu',
                         save_path: Optional[str] = None, depth=3) -> str:
    model_summary = torchinfo.summary(model, input_size=input_size, device=device, verbose=0, depth=depth)
    model_summary_str = str(model_summary)
    if save_path is not None:
        save_text_to_file(model_summary_str, save_path)
    return model_summary_str
