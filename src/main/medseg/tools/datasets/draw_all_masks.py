import os

import click
import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F

from medseg.config.config import load_and_parse_config
from medseg.data.dataset_manager import DatasetManager
from medseg.evaluation.segmentation_visualizer import SegmentationVisualizer
from medseg.util.img_ops import tensor_to_pil
from medseg.util.path_builder import PathBuilder, get_root_path

from medseg.data.transforms import torchvision_transforms as mt_torch


@click.command()
@click.option('--cfg_path', '--c', type=str, required=True, help='Path to a minimal config.')
@click.option('--out', '--o', type=str, required=False, help='Output path.')
@click.option('--enable_transforms', '--t', is_flag=True,
              help='Apply the transform pipeline in the config before calculating.')
@click.option('--use_contours', '--t', is_flag=True,
              help='Use contours instead of overlay.')
@click.option('--aux_only', '--t', is_flag=True, help='Skip the main dataset and use only aux datasets.')
def draw_all_masks(cfg_path: str, out: str, enable_transforms=False, use_contours=True, aux_only=True):
    cfg = load_and_parse_config(cfg_path)
    dataset_manager = DatasetManager(cfg)
    out_pb = PathBuilder.out_builder().add("GT_segs")

    # Build dict to iterate through
    split_ds_dict = {} if aux_only else dataset_manager.datasets

    if dataset_manager.has_aux_test_datasets():
        for ds, data_loader in dataset_manager.aux_test_datasets_and_loaders:
            split_ds_dict.update({f"AUX_{ds.get_name()}_{ds.split_type}": ds})

    for split, ds in split_ds_dict.items():
        for i in range(len(ds)):
            img, mask, real_i = ds.__getitem__(i) if enable_transforms else ds.load_img_mask(i)
            if enable_transforms:
                # mask = ds.class_mapping.revert_class_mapping(mask)

                # Denormalize for AUX datasets, such as UKW
                if isinstance(split, str):
                    for transform in ds.transforms_manager.get_transforms_with_types({mt_torch.Normalize}):
                        img, mask = transform.denormalize(img, mask)
                        if isinstance(img, np.ndarray): img = Tensor(img)
                        if isinstance(mask, np.ndarray): mask = Tensor(mask)

                img = img * 255
            if isinstance(img, Tensor):
                img = tensor_to_pil(img)
            if isinstance(mask, Tensor):
                mask = F.to_pil_image(mask.float())
            if not use_contours:
                gt_image = SegmentationVisualizer.draw_mask_overlay(img, mask, (0, 200, 128))
            else:
                gt_image = SegmentationVisualizer.draw_mask_contour(img, mask, (0, 200, 128), width=3)
            file_name = ds.images[i]
            split_name = split.value if not isinstance(split, str) else split.split(".")[1].lower()
            if out is not None:
                save_path = os.path.join(out, ds.get_name(), split_name, f"{file_name}_seg.png")
            else:
                save_path = out_pb.clone().add(ds.get_name()).add(split_name).add(f"{file_name}_seg.png").build()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            gt_image.save(save_path)


def draw_all_GT_masks_from_wseg():
    project_root = get_root_path()
    image_folder_path = os.path.join(project_root, "data/datasets/wseg/images")
    mask_folder_path = os.path.join(project_root, "data/datasets/wseg/masks")
    save_path = os.path.join(project_root, "data/datasets/wseg/gt_plots")
    os.makedirs(save_path,exist_ok=True)

    draw_all_masks_from_folder(image_folder_path, mask_folder_path, save_path)


def draw_all_GT_masks_from_ukw():
    project_root = get_root_path()
    image_folder_path = os.path.join(project_root, "data/datasets/ukw/images")
    mask_folder_path = os.path.join(project_root, "data/datasets/ukw/masks")
    save_path = os.path.join(project_root, "data/datasets/ukw/gt_plots")
    os.makedirs(save_path,exist_ok=True)

    df = pd.read_csv(os.path.join(project_root, "data/datasets/ukw/index.csv"))
    image_mask_dict = {row["image"]: row["mask"] for _, row in df.iterrows()}

    draw_all_masks_from_folder(image_folder_path, mask_folder_path, save_path,
                               overlay=False, image_mask_dict=image_mask_dict)


def draw_all_masks_from_folder(image_folder_path, mask_folder_path, save_path, overlay=True, image_mask_dict=None):
    # Loop through each image-mask pair
    for img_name in os.listdir(image_folder_path):
        # Load image and mask
        img_path = os.path.join(image_folder_path, img_name)
        if image_mask_dict is None:
            mask_path = os.path.join(mask_folder_path, img_name)
        else:
            mask_path = os.path.join(mask_folder_path, image_mask_dict[img_name])

        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        if overlay:
            gt_image = SegmentationVisualizer.draw_mask_overlay(image, mask, (0, 200, 128))
        else:
            gt_image = SegmentationVisualizer.draw_mask_contour(image, mask, (0, 200, 128), width=10)
        save_img_path = os.path.join(save_path, f"{img_name.split('.')[0]}_seg.png")
        gt_image.save(save_img_path)


if __name__ == '__main__':
    # draw_all_GT_masks_from_ukw()
    # draw_all_GT_masks_from_wseg()
    draw_all_masks()
