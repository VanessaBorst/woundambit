# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


from typing import Tuple, List, Callable

import numpy as np
from PIL.Image import Image
from torch import Tensor

from medseg.data.split_type import SplitType

from medseg.data.transforms import torchvision_transforms as mt_torch
from medseg.data.transforms.transforms import MedsegTransform


class TransformsManager:
    """
    TransformsManager class that manages and applies all relevant transform instances to the dataset.
    """

    def __init__(self, cfg: dict, split_type: SplitType,
                 default_hw: Tuple[int, int] = None, get_height_width: Callable = None):
        self.transforms: List[MedsegTransform] = []
        self.transforms_backend = cfg["settings"].get("transforms_backend", "torchvision")
        if "transforms" in cfg and split_type.value in cfg["transforms"]:
            for transform_entry in cfg["transforms"][split_type.value]:
                for transform_str, medseg_transform in transform_entry.items():
                    self.transforms.append(medseg_transform)
        # In the case of a torchvision backed, add a ToTensor transform at the beginning of the pipeline if it's not
        # already there
        if self.transforms_backend == "torchvision":
            if not self.contains_transform(mt_torch.ToTensor):
                self.transforms.insert(0, mt_torch.ToTensor({'normalize_masks': False, 'normalize_images': True}))
                print(
                    f"No ToTensor transform found in the {split_type.get_full_name().lower()} pipeline. Adding ToTensor "
                    f"transform at the beginning of the pipeline")

        self.default_hw = default_hw
        if self.default_hw is not None:
            assert len(self.default_hw) == 2
            assert self.default_hw[0] > 0 and self.default_hw[1] > 0

        self.get_height_width_func = get_height_width
        if default_hw is not None:
            self.set_transforms_height_width()

    def apply_transforms(self, img: Image, mask: Image, real_i: int) -> Tuple[Tensor, Tensor]:
        """
        Apply the determined transforms to the given image and mask.

        :param img: Input image
        :param mask: Input mask
        :param real_i: Real index of the image
        :return: Tuple containing the transformed image and mask
        """
        return self.apply_transforms_from_list(img, mask, real_i, self.transforms)

    def apply_transforms_from_list(self, img: Image, mask: Image, real_i: int, transforms_list: list):
        for medseg_transform in transforms_list:
            if self.default_hw is None and medseg_transform.needs_height_width:
                h, w = self.get_height_width_func(real_i)
                img, mask = medseg_transform.apply(img, mask, new_args={"h": h, "w": w}, replace_args=False)
            else:
                img, mask = medseg_transform.apply(img, mask)
        return img, mask

    def contains_transform(self, transform_type: type) -> bool:
        """
        Check if the transform pipeline contains a transform of the given type.

        :param transform_type: Type of the transform to check for
        :return: True if the transform pipeline contains a transform of the given type, False otherwise
        """
        for medseg_transform in self.transforms:
            if isinstance(medseg_transform, transform_type):
                return True
        return False

    def set_transforms_height_width(self):
        for medseg_transform in self.transforms:
            if medseg_transform.needs_height_width:
                args = {"h": self.default_hw[0], "w": self.default_hw[1]}
                medseg_transform.use_args(args)

    def set_args(self, args: dict, transform_type: type):
        for medseg_transform in self.transforms:
            if isinstance(medseg_transform, transform_type):
                medseg_transform.use_args(args)

    def get_transforms_before(self, transform: MedsegTransform) -> list:
        transforms_before = []
        for medseg_transform in self.transforms:
            if medseg_transform == transform:
                return transforms_before
            transforms_before.append(medseg_transform)
        return []

    def get_transforms_with_types(self, transform_types: set) -> list:
        transforms_with_types = []
        for medseg_transform in self.transforms:
            if any(isinstance(medseg_transform, t_type) for t_type in transform_types):
                transforms_with_types.append(medseg_transform)
        return transforms_with_types

    def get_transforms_without_types(self, transform_types: set) -> list:
        transforms_without_types = []
        for medseg_transform in self.transforms:
            if not any(isinstance(medseg_transform, t_type) for t_type in transform_types):
                transforms_without_types.append(medseg_transform)
        return transforms_without_types

    def get_pixel_range(self, for_mask: bool = False) -> Tuple[float, float]:
        min_pixel_value, max_pixel_value = 0, 255
        # TODO this could give wrong values for super weird transform pipelines
        #  (maybe disallow those in config check?)
        for medseg_transform in self.transforms:
            if isinstance(medseg_transform, mt_torch.Normalize):
                if (for_mask and medseg_transform.args["normalize_masks"]) or (
                        not for_mask and medseg_transform.args["normalize_images"]):
                    mean = medseg_transform.args["mean"]
                    std = medseg_transform.args["std"]

                    min_pixel_value = (min_pixel_value - mean) / std
                    max_pixel_value = (max_pixel_value - mean) / std

            elif isinstance(medseg_transform, mt_torch.ToTensor):
                if for_mask and medseg_transform.normalize_masks or not for_mask and medseg_transform.normalize_images:
                    min_pixel_value /= 255
                    max_pixel_value /= 255

        return min_pixel_value, max_pixel_value
