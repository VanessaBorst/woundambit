import os
from collections import OrderedDict
from typing import List, Tuple, Optional, Set, Dict

import numpy as np
import pandas as pd
from beartype import beartype

from medseg.data.converters.helpers import get_image_dims, create_dataset_subfolders, collect_and_sort_files, \
    check_matches, copy_all_files
from medseg.data.split_type import SplitType
from medseg.util.random import get_splits


@beartype
def create_dataset(paths_in: List[Tuple[str, str]],
                   out_path: str,
                   class_dict: Dict[str, int],
                   ratios: Dict[str, float],
                   seed: Optional[int] = 42,
                   exclude_files: Optional[Set[str]] = None,
                   copy_files: bool = True,
                   strict_filename_matching: bool = True
                   ) -> None:

    """ Creates a dataset with a train, validation, and test split according to the specified ratios. If the seed is
    None, the splits will occur according to an alphanumeric sorting of the filenames. Otherwise, the seed will be used
    to generate a randomized assignment of the files to the splits."""

    assert len(ratios.keys()) == 3
    assert ratios.keys() == {'train', 'val', 'test'}
    assert sum(ratios.values()) == 1.0

    all_images, all_masks = collect_and_sort_files(paths_in)

    if exclude_files is not None:
        all_images, all_masks = exclude_files_and_save_blacklist(all_images, all_masks, exclude_files, out_path)

    images_list = list(all_images.keys())
    masks_list = list(all_masks.keys())

    if strict_filename_matching:
        check_matches(images_list, masks_list)

    n_samples = len(all_images)
    indices = list(range(n_samples))
    train, val, test = get_splits(np.array(indices), train_ratio=ratios['train'], val_ratio=ratios['val'], seed=seed)

    create_index_classes_csv(all_images, all_masks, out_path, train.tolist(), val.tolist(), test.tolist(), class_dict)

    if copy_files:
        img_path, mask_path = create_dataset_subfolders(out_path)
        copy_all_files(paths_in, img_path, mask_path)


@beartype
def create_dataset_predefined_split(train_paths_in: List[Tuple[str, str]],
                                    val_paths_in: List[Tuple[str, str]],
                                    test_paths_in: List[Tuple[str, str]],
                                    out_path: str,
                                    class_dict: dict,
                                    exclude_files: Optional[Set[str]] = None,
                                    copy_files: bool = True,
                                    strict_filename_matching=True) -> None:
    """
    Creates a new dataset from a predefined split of train, validation, and test paths. It is not required to provide
    paths for every split, i.e. the validation and test paths can be empty and then the dataset will consist only of
    a train split.
    """
    train_images, train_masks = collect_and_sort_files(train_paths_in)
    val_images, val_masks = collect_and_sort_files(val_paths_in)
    test_images, test_masks = collect_and_sort_files(test_paths_in)

    all_images = OrderedDict(sorted({**train_images, **val_images, **test_images}.items()))
    all_masks = OrderedDict(sorted({**train_masks, **val_masks, **test_masks}.items()))

    if exclude_files is not None:
        all_images, all_masks = exclude_files_and_save_blacklist(all_images, all_masks, exclude_files, out_path)

    images_list = list(all_images.keys())
    masks_list = list(all_masks.keys())

    if strict_filename_matching:
        check_matches(images_list, masks_list)

    train = [images_list.index(img_name) for img_name in images_list if img_name in train_images.keys()]
    val = [images_list.index(img_name) for img_name in images_list if img_name in val_images.keys()]
    test = [images_list.index(img_name) for img_name in images_list if img_name in test_images.keys()]

    assert len(train) + len(val) + len(test) == len(all_images) == len(all_masks)

    create_index_classes_csv(all_images, all_masks, out_path, train, val, test, class_dict)

    if copy_files:
        img_path, mask_path = create_dataset_subfolders(out_path)
        copy_all_files(train_paths_in + val_paths_in + test_paths_in, img_path, mask_path)


@beartype
def check_duplicate_filenames(*image_dicts: OrderedDict[str, str]):
    all_img_set = set()
    for img_dict in image_dicts:
        img_set_keys = set(img_dict.keys())
        if not all_img_set.isdisjoint(img_set_keys):
            raise ValueError("Duplicate file names found between images of different sets.")
        all_img_set |= img_set_keys


@beartype
def create_index_classes_csv(all_images: OrderedDict[str, str],
                             all_masks: OrderedDict[str, str],
                             out_path: str,
                             train_indices: List[int],
                             val_indices: List[int],
                             test_indices: List[int],
                             class_dict: Dict[str, int]) -> None:
    assert len(all_images) == len(all_masks)
    n_samples = len(all_images)
    assert len(train_indices) + len(val_indices) + len(test_indices) == n_samples

    # create rows for index dataframe
    rows = []
    image_filenames = list(all_images.keys())
    mask_filenames = list(all_masks.keys())
    for i in range(n_samples):
        if i in train_indices:
            split = SplitType.TRAIN.value
        elif i in val_indices:
            split = SplitType.VAL.value
        elif i in test_indices:
            split = SplitType.TEST.value
        else:
            raise ValueError(f"Index {i} is not in train_indices, val_indices or test_indices")
        img_filename = image_filenames[i]
        mask_filename = mask_filenames[i]
        img_ext = os.path.splitext(img_filename)[1][1:]
        mask_ext = os.path.splitext(mask_filename)[1][1:]
        img_height, img_width = get_image_dims(all_images[img_filename], img_ext)
        mask_height, mask_width = get_image_dims(all_masks[mask_filename], mask_ext)
        if img_height != mask_height or img_width != mask_width:
            raise Warning(f"Image and mask dimensions do not match for file {img_filename}")
        rows.append([img_filename, mask_filename, split, img_height, img_width, img_ext, mask_ext])

    # create index dataframe and save to csv
    os.makedirs(out_path, exist_ok=True)
    df = pd.DataFrame(rows, columns=["image", "mask", "split", "height", "width", "img_type", "mask_type"])
    df.to_csv(os.path.join(out_path, "index.csv"), index=True)
    # convert class_dict to list of lists
    class_list = []
    for key, value in class_dict.items():
        class_list.append([key, value])
    df = pd.DataFrame(class_list, columns=["label", "pixel_value"])
    df.to_csv(os.path.join(out_path, "classes.csv"), index=True)


@beartype
def exclude_files_and_save_blacklist(all_images: OrderedDict[str, str],
                                     all_masks: OrderedDict[str, str],
                                     exclude_files: Set[str],
                                     out_path):
    """ Removes blacklisted files from the dataset and saves the blacklist to a csv file."""

    if exclude_files is not None:
        img_keys_to_remove, mask_keys_to_remove = set(), set()
        for i, file_name in enumerate(all_images.keys()):
            if file_name in exclude_files:
                img_keys_to_remove.add(file_name)
                mask_keys_to_remove.add(list(all_masks.keys())[i])

        for i, file_name in enumerate(all_masks.keys()):
            if file_name in exclude_files:
                mask_keys_to_remove.add(file_name)
                img_keys_to_remove.add(list(all_images.keys())[i])

        for key in img_keys_to_remove:
            all_images.pop(key)

        for key in mask_keys_to_remove:
            all_masks.pop(key)

        # save blacklist to csv
        excl_list = sorted(list(exclude_files))
        df = pd.DataFrame(excl_list, columns=["blacklist"])
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(os.path.join(out_path, "blacklist.csv"), index=True)
    return all_images, all_masks


