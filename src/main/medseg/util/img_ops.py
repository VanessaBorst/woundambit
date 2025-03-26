import random
from typing import Tuple, Callable, Union

import numpy as np
import skimage
import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import functional as F


def open_image(image_path: str, file_extension: str) -> Image:
    """
    Open an image and return it as PIL.
    :param image_path:      Path to the image.
    :param file_extension:  File extension of the image.
    :return:                Image as PIL.
    """
    if file_extension == "tiff" or file_extension == "tif":
        # PIL can produce exceptions reading some .tif/.tiff files, for example in the cvc-clinicdb dataset
        return Image.fromarray(skimage.io.imread(image_path))
    return Image.open(image_path)


def calculate_mean_std(get_image_func: Callable, num_images: int, num_channels: int) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    total_pixels = 0
    running_mean = torch.zeros(num_channels)
    running_mean_of_squares = torch.zeros(num_channels)

    for i in range(num_images):
        img = get_image_func(i)
        c, h, w = img.shape
        assert c == 1 or c == 3, f"Channel number should be 1 or 3, not {c}"
        img_pixels = h * w
        sum_ = torch.sum(img, dim=[1, 2])
        sum_of_square = torch.sum(img ** 2, dim=[1, 2])
        running_mean = (total_pixels * running_mean + sum_) / (total_pixels + img_pixels)
        running_mean_of_squares = (total_pixels * running_mean_of_squares + sum_of_square) / (total_pixels + img_pixels)

        total_pixels += img_pixels

    variance = running_mean_of_squares - running_mean ** 2
    std_deviation = torch.sqrt(variance)
    return running_mean, std_deviation


def get_pil_mode(n_channels):
    if n_channels == 1:
        return "L"
    elif n_channels == 3:
        return "RGB"
    else:
        raise ValueError("Unsupported number of channels: {}".format(n_channels))


def to_tensor(img: Union[PILImage, np.ndarray, torch.Tensor]) -> Tensor:
    if isinstance(img, PILImage):
        return pil_to_tensor(img)
    elif isinstance(img, np.ndarray):
        return numpy_to_tensor(img)
    return img


def pil_to_tensor(pil_img):
    return numpy_to_tensor(np.array(pil_img))


def tensor_to_pil(tensor_img: torch.Tensor):
    # TODO: does not work for tensor with 1 channel
    tensor_np = tensor_to_numpy(tensor_img)
    n_channels = tensor_np.shape[-1]
    pil_mode = get_pil_mode(n_channels)
    if n_channels == 1:
        tensor_np = np.squeeze(tensor_np, axis=-1)
    return Image.fromarray(tensor_np.astype(np.uint8), pil_mode)


def pil_to_numpy(image: PILImage) -> np.ndarray:
    return np.array(image)


def numpy_to_pil(image: np.ndarray) -> PILImage:
    return Image.fromarray(image)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if len(tensor.shape) <= 2:
        return tensor.numpy()
    elif tensor.shape[-3] < tensor.shape[-2] and tensor.shape[-3] < tensor.shape[-1]:
        # tensor is in the format (..., C, H, W) => convert it to (..., H, W, C)
        leading_dims = range(len(tensor.shape) - 3)
        tensor = tensor.permute(*leading_dims, -2, -1, -3)
    return tensor.numpy()


def numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image)
    if len(tensor.shape) <= 2:
        return tensor
    elif tensor.shape[-2] > tensor.shape[-1] and tensor.shape[-3] > tensor.shape[-1]:
        # tensor is in the format (..., H, W, C) => convert it to (..., C, H, W)
        leading_dims = range(len(tensor.shape) - 3)
        tensor = tensor.permute(*leading_dims, -1, -3, -2)
    return tensor


def calculate_square_padding(current_height: int, current_width: int) -> Tuple[int, int, int, int]:
    """
    Calculate the padding required to make an image square while maintaining its aspect ratio.

    :param current_height: The current height of the image
    :param current_width: The current width of the image
    :return: A tuple containing the padding values (pad_left, pad_top, pad_right, pad_bottom)
    """
    target_size = max(current_height, current_width)
    pad_top = (target_size - current_height) // 2
    pad_bottom = target_size - current_height - pad_top
    pad_left = (target_size - current_width) // 2
    pad_right = target_size - current_width - pad_left
    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return padding


def normalize_tensor(tensor: Tensor, current_range: Tuple[float, float], target_range: Tuple[float, float]) -> Tensor:
    """
    Normalize a PyTorch tensor from a current range of values to a target range of values.

    :param tensor: The input tensor to be normalized
    :param current_range: A tuple containing the current minimum and maximum values of the tensor
    :param target_range: A tuple containing the target minimum and maximum values for the normalized tensor
    :return: The normalized tensor
    """
    current_min, current_max = current_range
    target_min, target_max = target_range

    tensor = (tensor - current_min) / (current_max - current_min)
    tensor = tensor * (target_max - target_min) + target_min

    return tensor


def prediction_to_segmentation_mask(prediction: Tensor, current_range: Tuple[float, float],
                                    target_range: Tuple[float, float], is_multiclass: bool,
                                    channel_dim=1) -> Tensor:
    """
    Convert a prediction tensor to a segmentation mask

    :param prediction: The prediction tensor
    :return: The segmentation mask
    """
    if is_multiclass:
        prediction = logits_to_segmentation_mask(prediction, channel_dim)
    else:
        prediction = prediction > 0.5

    prediction = normalize_tensor(prediction, current_range, target_range)
    return prediction


def logits_to_segmentation_mask(logits: Tensor, channel_dim=1) -> Tensor:
    """
    Converts the logits to a segmentation mask by selecting the class with the highest probability for each pixel.
    :param logits: 4D or 3D tensor containing the logits for each image in the batch, pixel, and class
    :param channel_dim: dimension of the channel axis
    """
    return torch.argmax(logits, dim=channel_dim).float()


def multiscale(in_size: int, images: Tensor, masks: Tensor, multiscale_factor: float, divisor: int = 64,
               align_corners=False):
    new_size = random.randrange(int(in_size * (1 - multiscale_factor)),
                                int(in_size * (1 + multiscale_factor))) // divisor * divisor
    images = F.interpolate(images, size=(new_size, new_size), mode='bilinear', align_corners=align_corners)
    masks = F.interpolate(masks, size=(new_size, new_size), mode='bilinear', align_corners=align_corners)
    return images, masks


def denormalize_image(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean
