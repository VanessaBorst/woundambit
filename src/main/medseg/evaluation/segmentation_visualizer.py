import random
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage
from beartype import beartype
from skimage import measure
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode

from medseg.data.datasets.medseg_dataset import MedsegDataset

from medseg.data.transforms import torchvision_transforms as mt_torch
from medseg.data.transforms.transforms import TorchvisionTransform
from medseg.util.img_ops import calculate_square_padding, logits_to_segmentation_mask, numpy_to_tensor, \
    tensor_to_pil
from medseg.util.path_builder import PathBuilder


class ImageSaveMode(Enum):
    ALL = 'all'
    WORST = 'worst'
    RANDOM_SUBSET = 'random_subset'
    ENSEMBLE_EVAL = 'ensemble_eval'


class Color(Enum):
    CYAN = (0, 255, 255)
    GREEN = (0, 255, 0)
    INDIGO = (75, 0, 130)
    MAGENTA = (244, 10, 244)
    RED = (255, 0, 0)
    ORANGE = (255, 140, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)
    PINK = (255, 192, 203)
    BROWN = (165, 42, 42)
    LIME = (0, 255, 0)
    TEAL = (0, 128, 128)
    OLIVE = (128, 128, 0)
    NAVY = (0, 0, 128)
    MAROON = (128, 0, 0)
    WHITE = (255, 255, 255)


class SegmentationVisualizer:
    @beartype
    def __init__(self, model: nn.Module, device: torch.device, dataset: MedsegDataset, img_path_builder: PathBuilder,
                 img_size: int):
        self.model = model if not hasattr(model, "medcam_dict") else model.medcam_dict["orig_model"]
        self.dataset = dataset
        self.device = device
        self.class_info = self._create_class_info(dataset.get_class_defs())
        self.img_size = img_size
        self.img_path_builder = img_path_builder

    @staticmethod
    @beartype
    def _create_class_info(class_defs: List[Dict[str, any]]) -> Dict[int, Dict[str, Union[str, Tuple[int, int, int]]]]:
        # TODO: refactor
        class_info = {}
        colors = list(Color)
        class_defs_dict = {c["pixel_value"]: c["label"] for c in class_defs if c["label"] != "background"}
        for i, (pixel_value, class_label) in enumerate(class_defs_dict.items()):
            color = colors[i * 2].value if i < len(colors) else tuple(np.random.randint(0, 256, size=3))
            pred_color = colors[i * 2 + 1].value if i * 2 + 1 < len(colors) else tuple(
                np.random.randint(0, 256, size=3))

            class_info[pixel_value] = {
                "class": class_label,
                "color": color,
                "pred_color": pred_color,
            }
        return class_info

    @staticmethod
    @beartype
    def _convert_to_pil(img: Union[np.ndarray, PILImage, torch.Tensor]) -> PILImage:
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        elif isinstance(img, np.ndarray):
            img = F.to_pil_image(torch.from_numpy(img))
        return img

    @staticmethod
    @beartype
    def _draw_masks(
            img: PILImage,
            mask: Union[np.ndarray, PILImage, torch.Tensor],
            class_info: Dict[int, Dict[str, Union[str, Tuple[int, int, int]]]],
            is_label: bool,
            draw_contour: bool,
    ) -> PILImage:

        mask = SegmentationVisualizer._convert_to_pil(mask)
        unique_values = np.unique(np.array(mask))

        for class_number in unique_values:
            if class_number in class_info:
                color_key = "color" if is_label else "pred_color"
                color = class_info[class_number][color_key]
                mask_class = Image.fromarray((np.array(mask) == class_number).astype(np.uint8) * 255)
                if draw_contour:
                    img = SegmentationVisualizer.draw_mask_contour(img, mask_class, color)
                else:
                    img = SegmentationVisualizer.draw_mask_overlay(img, mask_class, color)
        return img

    @staticmethod
    @beartype
    def draw_mask_contour(img: PILImage, mask: PILImage, color: Tuple[int, int, int], width=2) -> PILImage:
        mask = np.array(mask)
        mask[mask > 0] = 255
        contours = measure.find_contours(mask, 0.5)
        draw = ImageDraw.Draw(img)
        for contour in contours:
            contour = [(p[1], p[0]) for p in contour]  # Convert (row, col) to (x, y)
            draw.line(contour + [contour[0]], fill=color, width=width)
        return img

    @staticmethod
    @beartype
    def draw_mask_overlay(img: PILImage, mask: PILImage, color: Tuple[int, int, int]) -> PILImage:

        img_rgba = img.convert("RGBA")
        mask_rgba = mask.convert("RGBA")
        threshold = 127
        alpha = 120
        mask_data = mask_rgba.getdata()
        new_mask_data = []

        for r, g, b, a in mask_data:
            if r <= threshold and g <= threshold and b <= threshold:
                new_mask_data.append((0, 0, 0, 0))
            else:
                new_mask_data.append((color[0], color[1], color[2], alpha))

        new_mask = Image.new("RGBA", mask_rgba.size)
        new_mask.putdata(new_mask_data)
        img_rgba.alpha_composite(new_mask)
        return img_rgba

    @staticmethod
    @beartype
    def draw_text(img: PILImage, text: str, color: Tuple[int, int, int],
                  position: Tuple[int, int]) -> PILImage:
        img_draw = ImageDraw.Draw(img)
        img_draw.text(position, text, fill=color)
        return img

    @beartype
    def get_seg_pred_label(
            self,
            img: Union[np.ndarray, PILImage, torch.Tensor],
            mask_pred: Union[np.ndarray, PILImage, torch.Tensor],
            mask_label: Union[np.ndarray, PILImage, torch.Tensor],
    ) -> PILImage:
        result = self._convert_to_pil(img)
        result = self._draw_masks(result, mask_pred, self.class_info, False, True)
        result = self._draw_masks(result, mask_label, self.class_info, True, False)
        return result

    @beartype
    def get_seg_pred(self, img: Union[np.ndarray, PILImage, torch.Tensor],
                     mask_pred: Union[np.ndarray, PILImage, torch.Tensor]) -> PILImage:
        result = self._convert_to_pil(img)
        result = self._draw_masks(result, mask_pred, self.class_info, False, False)
        return result

    @beartype
    def get_seg_label(self, img: Union[np.ndarray, PILImage, torch.Tensor],
                      mask_label: Union[np.ndarray, PILImage, torch.Tensor]) -> PILImage:
        result = self._convert_to_pil(img)
        result = self._draw_masks(result, mask_label, self.class_info, True, False)
        return result

    @beartype
    def save_image_pairs(self, indices: List[int], sample_prefix: str):
        for i in indices:
            image, mask, real_i = self.dataset.load_img_mask(i)

            image, mask = F.to_tensor(image), F.to_tensor(mask)
            target_size = [self.img_size, self.img_size]
            height, width = image.shape[-2:]
            padding = calculate_square_padding(height, width)
            mask_padding_fill = self.dataset.class_mapping.bg_pixel
            image = F.pad(image, list(padding))
            mask = F.pad(mask, list(padding), fill=mask_padding_fill)
            image = F.resize(image, target_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            mask = F.resize(mask, target_size, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
            _image, _mask = image, mask

            # look for transforms that can include normalization as model expects data within this distribution
            transforms_types_to_apply = {mt_torch.ToTensor, mt_torch.Normalize}
            transforms = self.dataset.transforms_manager.get_transforms_with_types(transforms_types_to_apply)

            # Convert to PIL to imitate normal transform pipeline
            _image = tensor_to_pil(_image)

            for transform in transforms:
                # some conversions, just in case
                if isinstance(transform, TorchvisionTransform):
                    if isinstance(_image, np.ndarray): _image = numpy_to_tensor(_image)
                _image, _ = transform.apply(_image, _mask)

            if isinstance(_image, np.ndarray): _image = numpy_to_tensor(_image)
            _image = image.unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(_image.to(device=self.device))
            prediction = prediction.to(device=torch.device("cpu"))
            prediction = logits_to_segmentation_mask(prediction) if self.dataset.is_multiclass() else prediction > 0.5
            img_prefix = f"{i:04d}"
            prediction = prediction.squeeze(0)
            self.save_segmentation_sample(sample_prefix, img_prefix, image, mask, prediction)

    @beartype
    def save_image_pairs_apply_transforms(self, indices: List[int], sample_prefix: str):
        for i in indices:
            img, mask, real_i = self.dataset.load_img_mask(i)
            t_manager = self.dataset.transforms_manager
            exclude_transforms = {mt_torch.ColorJitter, mt_torch.RandomPhotometricDistort}
            transforms_to_apply = t_manager.get_transforms_without_types(exclude_transforms)
            img, mask = t_manager.apply_transforms_from_list(img, mask, real_i, transforms_to_apply)

            self.model.eval()
            with torch.no_grad():
                pred = self.model(img.unsqueeze(0).to(device=self.device))
            pred = pred.cpu()
            pred = logits_to_segmentation_mask(pred).int() if self.dataset.is_multiclass() else pred > 0.5

            img, mask = self.revert_normalizations(img, mask)
            pred = pred.squeeze(0) if len(pred.shape) == 4 else pred
            # revert shapes to H, W, C
            #img = img.permute(1, 2, 0)
            #mask = mask.permute(1, 2, 0)
            #pred = pred.permute(1, 2, 0)
            # revert to a range of 0-255
            #img = img * 255
            img_prefix = f"{i:04d}"
            self.save_segmentation_sample(sample_prefix, img_prefix, img, mask, pred)

    def revert_normalizations(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        for transform in self.dataset.transforms_manager.get_transforms_with_types({mt_torch.Normalize}):
            img, mask = transform.denormalize(img, mask)
            if isinstance(img, np.ndarray): img = Tensor(img)
            if isinstance(mask, np.ndarray): mask = Tensor(mask)
        return img, mask

    def save_segmentation_sample(self, sample_prefix: str,
                                 img_prefix: str,
                                 image,
                                 mask,
                                 prediction,
                                 revert_mask_class_mapping=False):

        prediction = prediction.cpu()
        mask = mask.cpu()
        image = image.cpu()

        prediction = self.dataset.class_mapping.revert_class_mapping(prediction).int()
        if revert_mask_class_mapping:
            mask = self.dataset.class_mapping.revert_class_mapping(mask).int()
        # collapse mask to single channel
        # if prediction.shape[0] > 1:

        img_and_label_mask = self.get_seg_label(image, mask)
        img_and_pred_mask = self.get_seg_pred(image, prediction)
        label_save_path = self.img_path_builder.clone().add(sample_prefix).add(
            f"{sample_prefix}_{img_prefix}_label.jpg").build()
        pred_save_path = self.img_path_builder.clone().add(sample_prefix).add(
            f"{sample_prefix}_{img_prefix}_pred.jpg").build()
        img_and_label_mask.convert('RGB').save(label_save_path)
        img_and_pred_mask.convert('RGB').save(pred_save_path)

    @beartype
    def save_segmentations(self, worst_ids: Optional[List[str]] = None, n_random_samples: int = 0):
        images = self.dataset.images
        worst_images_i = []
        all_images_i = list(range(len(images)))

        if worst_ids is not None:
            for image_filename in worst_ids:
                if image_filename not in images:
                    print(f"Error in segmentation visualizer: Image {image_filename} is not in the dataset")
                    continue
                worst_images_i.append(images.index(image_filename))
            sample_prefix = "worst"
            self.save_image_pairs_apply_transforms(worst_images_i, sample_prefix)

        if n_random_samples > 0:
            # in case of both worst and random subset modes being active, sample from the set of images that are not
            # in the worst set
            sample_prefix = "random_subset"
            images_to_sample_i = set(all_images_i) - set(worst_images_i)
            n_images_to_sample = len(images_to_sample_i)
            n_random_samples = n_images_to_sample if n_random_samples > n_images_to_sample else n_random_samples
            random_subset_indices = random.sample(list(images_to_sample_i), n_random_samples)
            self.save_image_pairs_apply_transforms(random_subset_indices, sample_prefix)
