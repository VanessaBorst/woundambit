# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import abc
from copy import deepcopy
from typing import Union, Dict, Type

import numpy as np
import torch
from PIL.Image import Image
from torch import Tensor
from torch.nn import Module
from torchvision import transforms as t
from torchvision.transforms import v2 as t2, functional as F, ToPILImage, InterpolationMode

from medseg.util.img_ops import pil_to_numpy, numpy_to_pil

IMAGE_ONLY_TRANSFORMS = [
    t.Normalize,
    t.ColorJitter,
    t.GaussianBlur,
    t2.RandomPhotometricDistort
]

MASK_ONLY_TRANSFORMS = [
    'threshold',
]


class MedsegTransform(metaclass=abc.ABCMeta):
    """
    Abstract base class for all transforms.
    """

    def __init__(self,
                 transform_type: Union[Type[Module], Type[ToPILImage], str, None],
                 args: Dict,
                 backend: str):
        self.apply_to_mask = transform_type not in IMAGE_ONLY_TRANSFORMS
        self.apply_to_image = transform_type not in MASK_ONLY_TRANSFORMS
        self.type = transform_type
        self.args = args
        self.img_transforms = None
        self.mask_transforms = None
        self.composed = False
        self.needs_height_width = False
        self.backend = backend

    def __eq__(self, other):
        if not isinstance(other, MedsegTransform):
            return False
        is_equal = self.type == other.type and self.args == other.args
        is_equal = is_equal and self.apply_to_image == other.apply_to_image
        is_equal = is_equal and self.apply_to_mask == other.apply_to_mask
        is_equal = is_equal and self.backend == other.backend
        is_equal = is_equal and self.needs_height_width == other.needs_height_width
        return is_equal

    def compose(self):
        """
        Initializes and composes the transform with the provided arguments.
        """
        if 'apply_to_mask' in self.args: self.apply_to_mask = self.args.pop('apply_to_mask')
        if 'apply_to_image' in self.args: self.apply_to_image = self.args.pop('apply_to_image')
        built_transform = self.type(**self.args)
        if self.apply_to_image:
            self.img_transforms = t.Compose([built_transform])
        if self.apply_to_mask:
            self.mask_transforms = t.Compose([built_transform])
        self.composed = True

    def use_args(self, new_args: Dict, replace_args: bool = False):
        if replace_args or self.args is None:
            self.args = new_args
        else:
            self.args.update(new_args)

    def apply(self, img, mask, new_args=None, replace_args=False, temp_args=None):
        """
        Applies the transform to the image and mask. If new_args are provided, the transform is rebuilt with the new
        arguments.
        :param img: The image to transform
        :param mask: The mask to transform
        :param new_args: Optional new arguments to rebuild the transform with
        :param replace_args: If True, the new_args will replace the current args. If False, the new_args will be merged
        :param temp_args: Optional temporary arguments where the previous arguments are restored after applying
        :return: The transformed image and mask
        """
        previous_args = None
        if new_args is not None:
            if temp_args is not None:
                raise ValueError('Cannot provide both new_args and temp_args')
            self.use_args(new_args, replace_args)
            self.compose()
        if temp_args is not None:
            previous_args = deepcopy(self.args)
            self.use_args(temp_args, replace_args)
            self.compose()
        if not self.composed:
            self.compose()
        if self.apply_to_mask and self.apply_to_image:
            stored_rng_state = torch.get_rng_state()
            img = self.img_transforms(img)
            torch.set_rng_state(stored_rng_state)  # restore rng state to get same random transforms on mask
            mask = self.mask_transforms(mask)
            return img, mask
        else:
            if self.apply_to_image:
                img = self.img_transforms(img)
            if self.apply_to_mask:
                mask = self.mask_transforms(mask)
        if previous_args is not None:
            self.use_args(previous_args, True)
            self.compose()
        return img, mask


class TorchvisionTransform(MedsegTransform):
    def __init__(self, transform_type: Union[Type[Module], str, None], args: Dict):
        super().__init__(transform_type, args, backend='torchvision')


class Threshold(MedsegTransform):
    def __init__(self, args: dict, backend: str = "torchvision"):
        super().__init__('threshold', args, backend=backend)
        self.threshold = self.args.get("threshold", 0.5)
        self.pixel_min = self.args.get("pixel_min", 0)
        self.pixel_max = self.args.get("pixel_max", 1)
        self.img_transforms = self.apply_threshold
        self.mask_transforms = self.apply_threshold

    def apply_threshold(self, img_or_mask):
        is_pil = isinstance(img_or_mask, Image)
        is_numpy = isinstance(img_or_mask, np.ndarray)

        if is_pil:
            array = pil_to_numpy(img_or_mask)
        elif is_numpy:
            array = img_or_mask
        else:
            tensor = img_or_mask
            array = tensor.numpy()

        thresholded_array = np.where(array >= self.threshold, self.pixel_max, self.pixel_min).astype(array.dtype)

        if is_pil:
            return numpy_to_pil(thresholded_array)
        elif is_numpy:
            return thresholded_array
        else:
            return torch.from_numpy(thresholded_array)

    def compose(self):
        if 'apply_to_mask' in self.args: self.apply_to_mask = self.args.pop('apply_to_mask')
        if 'apply_to_image' in self.args: self.apply_to_image = self.args.pop('apply_to_image')


class RandomRatioResize(MedsegTransform):
    def __init__(self, args: dict, backend: str = "torchvision"):
        super().__init__('random_ratio_resize', args, backend=backend)
        self.needs_height_width = True  # causes transforms manager to inject current h and w params from loaded data
        self.min_ratio = self.args.pop("min_ratio", 0.5)
        self.max_ratio = self.args.pop("max_ratio", 2.0)
        assert self.min_ratio < self.max_ratio, "min_ratio must be less than max_ratio"
        self.img_transforms = self.apply_random_resize_to_img
        self.mask_transforms = self.apply_random_resize_to_mask
        self.img_args = None
        self.mask_args = None

    def __eq__(self, other):
        return super().__eq__(other) and self.min_ratio == other.min_ratio and self.max_ratio == other.max_ratio

    def compose(self):
        """
        Overrides the compose method of the base class.
        """
        args_copy = self.args.copy()
        if 'h' in args_copy.keys():
            args_copy.pop('h')
        if 'w' in args_copy.keys():
            args_copy.pop('w')
        mask_interpolation = InterpolationMode.NEAREST
        img_interpolation = InterpolationMode.BILINEAR
        img_antialias = True
        mask_antialias = False
        if "mask_interpolation" in args_copy.keys():
            mask_interpolation = args_copy.pop("mask_interpolation")
        if "interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("interpolation")
        if "img_interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("img_interpolation")
        img_args = {"interpolation": img_interpolation}
        mask_args = {"interpolation": mask_interpolation}
        if "antialias" in args_copy.keys():
            del args_copy["antialias"]
        img_args["antialias"] = args_copy.pop("img_antialias", img_antialias)
        mask_args["antialias"] = args_copy.pop("mask_antialias", mask_antialias)
        self.img_args = {**args_copy, **img_args}
        self.mask_args = {**args_copy, **mask_args}
        self.composed = True

    def apply_random_resize_to_img(self, img) -> Tensor:
        return self.apply_random_resize(img, is_mask=False)

    def apply_random_resize_to_mask(self, mask) -> Tensor:
        return self.apply_random_resize(mask, is_mask=True)

    def apply_random_resize(self, img_or_mask: Tensor, is_mask: bool) -> Tensor:
        is_pil = isinstance(img_or_mask, Image)
        is_numpy = isinstance(img_or_mask, np.ndarray)
        assert not is_pil, "PIL images are currently not supported for the RandomRatioResize transform. Please use " \
                           "ToTensor before"
        assert not is_numpy, "Numpy arrays are not supported for the RandomRatioResize transform. Please use " \
                             "ToTensor before"
        # random state is restored between image and mask calls by the inherited apply function
        ratio = torch.rand(1).item() * (self.max_ratio - self.min_ratio) + self.min_ratio
        # h and w are injected through the transforms manager
        new_height = int(self.args['h'] * ratio)
        new_width = int(self.args['w'] * ratio)
        resize_args = self.mask_args if is_mask else self.img_args
        img_or_mask = F.resize(img_or_mask, [new_height, new_width], **resize_args)
        return img_or_mask







