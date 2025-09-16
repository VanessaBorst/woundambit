# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
from torch import Tensor
from torchvision import transforms as t
from torchvision.transforms import InterpolationMode, v2 as t2

from medseg.data.transforms.transforms import TorchvisionTransform
from medseg.util.img_ops import calculate_square_padding, to_tensor


class ColorJitter(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.ColorJitter, args)


class GaussianBlur(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.GaussianBlur, args)


class Normalize(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.Normalize, args)

    def denormalize(self, img: Tensor, mask: Tensor):
        # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
        mean = self.args['mean']
        std = self.args['std']
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        temp_args = {'mean': mean_inv.tolist(), 'std': std_inv.tolist()}
        return self.apply(img, mask, temp_args=temp_args)

class Pad(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.Pad, args)
        self.needs_height_width = True

    def compose(self):

        if "w" not in self.args and "h" not in self.args:
            raise ValueError("Missing height and width arguments for padding transform")
        if 'size' not in self.args:
            raise ValueError("Missing size argument for resize_and_pad transform")

        pad_top = (self.args['size'] - self.args["h"]) // 2
        pad_bottom = self.args['size'] - self.args["h"] - pad_top
        pad_left = (self.args['size'] - self.args["w"]) // 2
        pad_right = self.args['size'] - self.args["w"] - pad_left
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        padding_mode = self.args.get('padding_mode', 'constant')
        fill = self.args.get('fill', 0)
        img_fill = self.args.get('img_fill', fill)
        mask_fill = self.args.get('mask_fill', fill)
        self.img_transforms = [t.Pad(padding, fill=img_fill, padding_mode=padding_mode)]
        self.mask_transforms = [t.Pad(padding, fill=mask_fill, padding_mode="constant")]


class PadAndResize(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(None, args)
        self.needs_height_width = True

    def compose(self):
        if "w" not in self.args and "h" not in self.args:
            raise ValueError("Missing height and width arguments for resize_and_pad transform")
        if 'size' not in self.args:
            raise ValueError("Missing size argument for resize_and_pad transform")
        target_dim = (self.args['size'], self.args['size'])
        padding = calculate_square_padding(self.args["h"], self.args["w"])
        padding_mode = self.args["padding_mode"] if "padding_mode" in self.args else 'constant'
        fill = self.args["fill"] if "fill" in self.args else 0
        mask_fill = self.args["mask_fill"] if "mask_fill" in self.args else 0
        img_interpolation = self.args.get("interpolation", InterpolationMode.BILINEAR)
        img_interpolation = self.args.get("img_interpolation", img_interpolation)
        mask_interpolation = self.args.get("mask_interpolation", InterpolationMode.NEAREST)
        pad_transform_img = t.Pad(padding, fill=fill, padding_mode=padding_mode)
        pad_transform_mask = t.Pad(padding, fill=mask_fill, padding_mode="constant")  # mask padding is always constant
        img_antialias = self.args.get('img_antialias', True)
        mask_antialias = self.args.get('mask_antialias', True)
        resize_transform_img = t.Resize(target_dim, interpolation=img_interpolation, antialias=img_antialias)
        # use nearest neighbour masks, as otherwise the interpolation would significantly alter the pixel values in
        # the mask (they correspond to class indices, so e.g. a border between the two classes 255 and 0 would be
        # interpolated / anti-aliased to values between 255 and 0). Antialiasing does not have a negative effect,
        # but will not work with nearest interpolation when the input is a tensor. In the case of a PIL image,
        # antialiasing cannot be turned off, so not sure if it will cause trouble here.
        # TODO: check if antialiasing causes exception if the input is Tensor and interpolation is nearest
        resize_transform_mask = t.Resize(target_dim, interpolation=mask_interpolation, antialias=mask_antialias)
        self.img_transforms = t.Compose([pad_transform_img, resize_transform_img])
        self.mask_transforms = t.Compose([pad_transform_mask, resize_transform_mask])


class RandomAffine(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.RandomAffine, args)

    def compose(self):
        """
        Overrides the compose method of the base class.
        """

        args_copy = self.args.copy()
        mask_fill = 0
        mask_interpolation = InterpolationMode.NEAREST
        img_interpolation = InterpolationMode.BILINEAR
        if "mask_interpolation" in args_copy.keys():
            mask_interpolation = args_copy.pop("mask_interpolation")
        if "interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("interpolation")
        if "img_interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("img_interpolation")
        if "mask_fill" in args_copy.keys():
            mask_fill = args_copy.pop("mask_fill")

        img_args = {"interpolation": img_interpolation}
        mask_args = {"fill": mask_fill, "interpolation": mask_interpolation}
        img_args = {**args_copy, **img_args}
        mask_args = {**args_copy, **mask_args}
        self.img_transforms = t.Compose([self.type(**img_args)])
        self.mask_transforms = t.Compose([self.type(**mask_args)])
        self.composed = True


class RandomCrop(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.RandomCrop, args)

    def compose(self):
        args_copy = self.args.copy()
        img_fill = args_copy.pop("fill") if "fill" in self.args else 0
        img_fill = args_copy.pop("img_fill") if "img_fill" in self.args else img_fill
        mask_fill = args_copy.pop("mask_fill") if "mask_fill" in self.args else img_fill
        img_transform = self.type(**args_copy, fill=img_fill)
        mask_transform = self.type(**args_copy, fill=mask_fill)
        self.img_transforms = t.Compose([img_transform])
        self.mask_transforms = t.Compose([mask_transform])


class RandomHorizontalFlip(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.RandomHorizontalFlip, args)


class RandomVerticalFlip(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.RandomVerticalFlip, args)


class RandomRotation(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.RandomRotation, args)

    def compose(self):
        """
        Overrides the compose method of the base class.
        """
        args_copy = self.args.copy()
        mask_fill = 0
        mask_interpolation = InterpolationMode.NEAREST
        img_interpolation = InterpolationMode.BILINEAR
        if "mask_interpolation" in args_copy.keys():
            mask_interpolation = args_copy.pop("mask_interpolation")
        if "interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("interpolation")
        if "img_interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("img_interpolation")
        if "mask_fill" in args_copy.keys():
            mask_fill = args_copy.pop("mask_fill")

        img_args = {"interpolation": img_interpolation}
        mask_args = {"fill": mask_fill, "interpolation": mask_interpolation}
        img_args = {**args_copy, **img_args}
        mask_args = {**args_copy, **mask_args}
        self.img_transforms = t.Compose([self.type(**img_args)])
        self.mask_transforms = t.Compose([self.type(**mask_args)])
        self.composed = True


class Resize(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.Resize, args)

    def compose(self):
        """
        Overrides the compose method of the base class.
        """
        args_copy = self.args.copy()
        mask_interpolation = InterpolationMode.NEAREST
        img_interpolation = InterpolationMode.BILINEAR
        if "mask_interpolation" in args_copy.keys():
            mask_interpolation = args_copy.pop("mask_interpolation")
        if "interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("interpolation")
        if "img_interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("img_interpolation")
        if "antialias" not in args_copy.keys():
            # set to default value. if it is not passed, torchvision puts out annoying warnings
            args_copy["antialias"] = True
        img_args = {"interpolation": img_interpolation}
        mask_args = {"interpolation": mask_interpolation}
        img_args = {**args_copy, **img_args}
        mask_args = {**args_copy, **mask_args}
        self.img_transforms = t.Compose([self.type(**img_args)])
        self.mask_transforms = t.Compose([self.type(**mask_args)])
        self.composed = True


class ToTensor(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.ToTensor, args)
        self.normalize_masks = self.args.get("normalize_masks", False)
        self.normalize_images = self.args.get("normalize_images", True)

    def compose(self):
        """
        Overrides the compose method of the base class to avoid using the ToTensor transform for the mask.
        """
        if self.normalize_images:
            self.img_transforms = t.Compose([self.type()])
        if self.normalize_masks:
            self.mask_transforms = t.Compose([self.type()])

    def apply(self, img, mask, new_args=None, replace_args=False, temp_args=None):
        """
        Overrides the apply method of the base class to not convert the mask to a range of [0, 1].
        """
        if new_args is not None:
            self.use_args(new_args, replace_args)
            self.compose()
        if not self.composed:
            self.compose()

        img = self.img_transforms(img) if self.normalize_images else to_tensor(img)
        mask = self.mask_transforms(mask) if self.normalize_masks else to_tensor(mask)

        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        if len(img.shape) < 3:
            img = img.unsqueeze(0)
        return img, mask


class ToPILImage(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.ToPILImage, args)


class RandomPhotometricDistort(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t2.RandomPhotometricDistort, args)


class RandomResizedCrop(TorchvisionTransform):
    def __init__(self, args: dict):
        super().__init__(t.RandomResizedCrop, args)

    def compose(self):
        """
        Overrides the compose method of the base class.
        """
        args_copy = self.args.copy()
        mask_interpolation = InterpolationMode.NEAREST
        img_interpolation = InterpolationMode.BILINEAR
        if "mask_interpolation" in args_copy.keys():
            mask_interpolation = args_copy.pop("mask_interpolation")
        if "interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("interpolation")
        if "img_interpolation" in args_copy.keys():
            img_interpolation = args_copy.pop("img_interpolation")
        img_args = {"interpolation": img_interpolation}
        mask_args = {"interpolation": mask_interpolation}
        img_args = {**args_copy, **img_args}
        mask_args = {**args_copy, **mask_args}
        self.img_transforms = t.Compose([self.type(**img_args)])
        self.mask_transforms = t.Compose([self.type(**mask_args)])
        self.composed = True