# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional

from medseg.models import segmentors
from medseg.models.segmentors.segmentor import Segmentor
from medseg.util.class_ops import get_class


def build_model(cfg: dict, out_channels: Optional[int] = None) -> Segmentor:
    """
    Build a segmentor from a config dict.
    Args:
        cfg: Config dict.
    Returns:
        The created segmentor.
    """
    segmentor_name = cfg['architecture']['arch_type']
    segmentor_class = get_class(segmentors, segmentor_name)
    assert segmentor_class is not None
    kwargs = {}
    if out_channels is not None:
        kwargs['out_channels'] = out_channels

    segmentor = segmentor_class.build_from_cfg(cfg, **kwargs)
    return segmentor
