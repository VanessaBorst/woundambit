import os
import shutil
from collections import OrderedDict
from typing import List, Tuple

from beartype import beartype

from medseg.util.img_ops import open_image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".PNG", ".JPG", ".JPEG", ".TIFF", ".TIF", ".BMP")


@beartype
def check_in_folder_paths(*folder_paths: str) -> None:
    for folder in folder_paths:
        if not os.path.exists(folder):
            raise ValueError(f"Unexpected input folder structure, can't find {folder}")


@beartype
def get_image_dims(img_path: str, extension: str) -> Tuple[int, int]:
    img = open_image(img_path, extension)
    return img.height, img.width


@beartype
def copy_files_from_to(from_path: str, to_path: str) -> None:
    if os.path.isfile(from_path):
        shutil.copy(from_path, to_path)
    elif os.path.isdir(from_path):
        for file in os.listdir(from_path):
            shutil.copy(os.path.join(from_path, file), os.path.join(to_path, file))
    else:
        raise ValueError("Invalid from_path. Must be a file or a directory.")


@beartype
def create_dataset_subfolders(out_path: str) -> Tuple[str, str]:
    img_path_out = os.path.join(out_path, "images")
    mask_path_out = os.path.join(out_path, "masks")
    os.makedirs(img_path_out, exist_ok=True)
    os.makedirs(mask_path_out, exist_ok=True)
    return img_path_out, mask_path_out


@beartype
def collect_and_sort_files(paths: List[Tuple[str, str]]) -> Tuple[OrderedDict[str, str], OrderedDict[str, str]]:
    images = OrderedDict()
    masks = OrderedDict()
    for img_path, mask_path in paths:
        if os.path.isdir(img_path):
            img_files = sorted(os.listdir(img_path))
            for file_name in img_files:
                if not file_name.endswith(IMAGE_EXTENSIONS): continue
                if file_name in images:
                    raise ValueError(f"Duplicate file name {file_name} in {img_path}")
                images[file_name] = os.path.join(img_path, file_name)
        else:
            images[os.path.basename(img_path)] = img_path
        if os.path.isdir(mask_path):
            mask_files = sorted(os.listdir(mask_path))
            for file_name in mask_files:
                if not file_name.endswith(IMAGE_EXTENSIONS): continue
                if file_name in masks:
                    raise ValueError(f"Duplicate file name {file_name} in {mask_path}")
                masks[file_name] = os.path.join(mask_path, file_name)
        else:
            masks[os.path.basename(mask_path)] = mask_path

    # Sort again then return. PyCharm displays false warnings here, which are disabled with the comments below
    # noinspection PyTypeChecker
    images = OrderedDict(sorted(images.items()))
    # noinspection PyTypeChecker
    masks = OrderedDict(sorted(masks.items()))
    return images, masks


@beartype
def check_matches(image_files: List[str], mask_files: List[str]) -> None:
    img_base_names = [os.path.splitext(img_file)[0] for img_file in image_files]
    mask_base_names = [os.path.splitext(mask_file)[0] for mask_file in mask_files]
    err_msg = ""
    images_without_match = ""
    masks_without_match = ""
    index_mismatches_imgs = 0
    index_mismatches_masks = 0

    for img_base_name in img_base_names:
        if img_base_name not in mask_base_names:
            images_without_match += f"{img_base_name}, "
        elif img_base_names.index(img_base_name) != mask_base_names.index(img_base_name):
            index_mismatches_imgs += 1

    for mask_base_name in mask_base_names:
        if mask_base_name not in img_base_names:
            masks_without_match += f"{mask_base_name}, "
        elif mask_base_names.index(mask_base_name) != img_base_names.index(mask_base_name):
            index_mismatches_masks += 1

    if index_mismatches_imgs > 0:
        err_msg += f"Number of image to mask mismatches (filenames are not equal) found: {index_mismatches_imgs}\n"
    if index_mismatches_masks > 0:
        err_msg += f"Number of mask to image mismatches (filenames are not equal) found: {index_mismatches_masks}\n"

    if len(images_without_match) > 0:
        err_msg += f"No masks found for image files: {images_without_match}\n"
    if len(masks_without_match) > 0:
        err_msg += f"No images found for mask files: {masks_without_match}\n"

    if len(err_msg) > 0:
        raise ValueError(err_msg)


@beartype
def copy_all_files(paths_in: List[Tuple[str, str]], img_path_out: str, mask_path_out: str) -> None:
    for img_path, mask_path in paths_in:
        copy_files_from_to(img_path, img_path_out)
        copy_files_from_to(mask_path, mask_path_out)
