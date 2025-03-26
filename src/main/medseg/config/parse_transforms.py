import re
from typing import Union

from medseg.data.split_type import SplitType
from medseg.data.transforms import torchvision_transforms as mt_torch
from medseg.data.transforms import transforms as mt

TORCHVISION_TRANSFORM_MAPPINGS = {
    'color_jitter': mt_torch.ColorJitter,
    'gaussian_blur': mt_torch.GaussianBlur,
    'normalize': mt_torch.Normalize,
    'to_tensor': mt_torch.ToTensor,
    'to_pil_image': mt_torch.ToPILImage,
    'pad': mt_torch.Pad,
    'pad_and_resize': mt_torch.PadAndResize,
    'random_affine': mt_torch.RandomAffine,
    'random_crop': mt_torch.RandomCrop,
    'random_horizontal_flip': mt_torch.RandomHorizontalFlip,
    'random_photometric_distort': mt_torch.RandomPhotometricDistort,
    'random_resized_crop': mt_torch.RandomResizedCrop,
    'random_rotation': mt_torch.RandomRotation,
    'random_vertical_flip': mt_torch.RandomVerticalFlip,
    'resize': mt_torch.Resize,
}

UNIVERSAL_TRANSFORM_MAPPINGS = {
    'threshold': mt.Threshold,
    'to_tensor': mt_torch.ToTensor,
    'random_ratio_resize': mt.RandomRatioResize,

}


TRANSFORMS_WITH_NESTED_TRANSFORMS = [
    'one_of',
]
NO_LOWER_CASE = [
    'model_name',
    'path'
]


def parse_transform(transform_key: str, transform_args: Union[dict, list], transforms_backend: str):
    mappings = UNIVERSAL_TRANSFORM_MAPPINGS
    if transforms_backend == 'torchvision':
        mappings.update(TORCHVISION_TRANSFORM_MAPPINGS)

    # Special slightly complicated case for nested transforms like one_of
    if transform_key in TRANSFORMS_WITH_NESTED_TRANSFORMS:
        if not isinstance(transform_args, list):
            raise ValueError(
                f"Transform parsing error: {transform_key} expects a list of nested transforms, or arguments, found "
                f"{transform_args}.")
        nested_transforms = []
        nested_args = {}

        for nested_transform_or_arg in transform_args:
            if not isinstance(nested_transform_or_arg, dict):
                raise ValueError(
                    f"Transform parsing error: {transform_key} expects a list of nested transforms or a relevant "
                    f"arguments dict, instead found {nested_transform_or_arg}.")
            for k, v in nested_transform_or_arg.items():
                if k in mappings.keys():
                    nested_transforms.append(mappings[k](v))
                else:
                    nested_args[k] = v
        parent_transform_class = mappings[transform_key]
        return parent_transform_class(nested_args, nested_transforms)

    elif transform_key in mappings.keys():
        return mappings[transform_key](transform_args)

    else:
        raise ValueError(
            f"Could not map {transform_key} from config to a transform with the given transforms backend {transforms_backend}.")


def parse_transforms(cfg: dict) -> dict:
    """
    Parses transforms in the config from strings to torch.nn.Module classes according to transform_mappings
    :param cfg the config dictionary
    :return the parsed config dictionary
    """

    if "transforms" not in cfg:
        return cfg
    cfg["transforms"] = parse_to_lowercase(cfg["transforms"])
    transforms_backend = cfg["settings"].get("transforms_backend", "torchvision")
    for split_key in [SplitType.TRAIN.value, SplitType.VAL.value, SplitType.TEST.value]:
        if split_key in cfg["transforms"] and cfg["transforms"][split_key] is False:
            # if the split key is defined and set to false, no transforms are applied, so it's set to an empty list
            cfg["transforms"][split_key] = []
        elif split_key not in cfg["transforms"]:
            cfg["transforms"][split_key] = []
    for split_type_key, split_transform_list in cfg["transforms"].items():
        if isinstance(split_transform_list, list):
            remove_items = []
            for i, transform_item in enumerate(split_transform_list):
                if isinstance(transform_item, dict):
                    for transform_key, transform_args in transform_item.items():
                        if isinstance(transform_args, bool):
                            if transform_args is False:
                                remove_items.append(i)
                                continue
                            else:
                                transform_args = {}
                        cfg["transforms"][split_type_key][i][transform_key] = parse_transform(transform_key,
                                                                                              transform_args,
                                                                                              transforms_backend)
                elif isinstance(transform_item, str):
                    cfg["transforms"][split_type_key][i] = {
                        transform_item: parse_transform(transform_item, {}, transforms_backend)}
            for r in remove_items:
                del split_transform_list[r]
    return cfg


def parse_to_lowercase(cfg: Union[dict, list]) -> Union[dict, list]:
    """
    Converts all keys and values in the config to lowercase.
     :param cfg the config dictionary or list
     :return the converted config dictionary or list
    """
    if isinstance(cfg, list):
        for i in range(len(cfg)):
            if isinstance(cfg[i], dict) or isinstance(cfg[i], list):
                cfg[i] = parse_to_lowercase(cfg[i])
            elif isinstance(cfg[i], str):
                # replace list item with lowercase and add an underscore before each capital letter
                parse_string_to_lowercase(cfg[i])

    if isinstance(cfg, dict):
        # loop over a copy of the keys to avoid key exception when changing the keys
        for key in list(cfg.keys()):
            old_key = key
            if isinstance(key, str):
                key = parse_string_to_lowercase(key)
                cfg[key] = cfg.pop(old_key)
            if isinstance(cfg[key], dict) or isinstance(cfg[key], list):
                cfg[key] = parse_to_lowercase(cfg[key])
            elif isinstance(cfg[key], str) and key not in NO_LOWER_CASE:
                # replace value with lowercase and add an underscore before each capital letter
                cfg[key] = parse_string_to_lowercase(cfg[key])
    return cfg


def parse_string_to_lowercase(string: str) -> str:
    """
    Converts a string to lowercase and inserts underscores in front of previously capitalized letters.
    :param string: The string to convert
    :return: The converted string
    """
    # insert an underscore before each capital letter, but only if it's not a sequence of capital letters
    parsed_str = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', string)
    parsed_str = re.sub(r'([a-z])([A-Z])', r'\1_\2', parsed_str).lower()
    return parsed_str
