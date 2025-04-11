import os

import click
from beartype import beartype
from typing import List, Tuple
from pathlib import Path
import imagehash
import shutil
from fnmatch import fnmatch
from medseg.data.converters.converters import create_dataset
from medseg.util.img_ops import open_image
from medseg.data.converters.helpers import check_in_folder_paths


SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}

@beartype
def _find_similar(paths: List[str], cutoff=11, display_full_paths: bool = False) -> Tuple[List, List, List]:
    """
    Find potential duplicates in a list of folders containing images by calculating and comparing the image hashes.

    Args:
        paths (List[str]): A list of directory paths containing images to be compared for similarity OR a list of
                          direct paths to image files.
        cutoff (int, optional): The cutoff value for considering two images similar based on their hash difference.
                                The lower the cutoff value, the higher the similarity required. Default is 8.
        display_full_paths (bool, optional): Flag that determines if the full path should be displayed in the final
                                             evaluation. Can be helpful for comparing files from different folders or
                                             with identical filenames. Defaults to False
    Returns:
        Tuple[list, list, list]: A tuple containing three lists:
            1. identical_bytes (list): A list of lists, where each inner list contains filenames of images
                                       that have identical bytes.
            2. identical_hashes (list): A list of lists, where each inner list contains filenames of images
                                         that have identical hashes but different bytes.
            3. high_similarity (list): A list of lists, where each inner list contains filenames of images
                                        that have similar hashes based on the provided cutoff value.

    Example:
        identical_bytes, identical_hashes, high_similarity = find_similar(["path/to/folder1", "path/to/folder2"])
    """

    image_paths = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower()[1:] in SUPPORTED_EXTENSIONS:
            image_paths.append(path)
        elif path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                image_paths.extend([p for p in path.rglob(f"*.*") if fnmatch(p.suffix.lower(), f".{ext.lower()}")])

    # phash seems to be the most accurate method according to this test:
    # https://content-blockchain.org/research/testing-different-image-hash-functions/
    image_hashes = {str(image_path): imagehash.phash(open_image(str(image_path), image_path.suffix[1:])) for image_path
                    in image_paths}
    identical_bytes = {}
    identical_hashes = {}
    high_similarity = []
    for image_path_1, image_hash_1 in image_hashes.items():
        for image_path_2, image_hash_2 in image_hashes.items():
            if image_path_1 != image_path_2:
                img_name_1 = os.path.basename(image_path_1) if not display_full_paths else image_path_1
                img_name_2 = os.path.basename(image_path_2) if not display_full_paths else image_path_2
                # compare the hashes
                if image_hash_1 == image_hash_2:
                    # open the images and compare the bytes
                    with open(image_path_1, "rb") as f1, open(image_path_2, "rb") as f2:
                        if f1.read() == f2.read():
                            # add to set if it exists already, otherwise create new
                            if image_hash_1 in identical_bytes:
                                identical_bytes[image_hash_1].add(img_name_1)
                                identical_bytes[image_hash_1].add(img_name_2)
                            else:
                                identical_bytes[image_hash_1] = {img_name_1, img_name_2}
                        else:
                            # add to set if it exists already, otherwise create new
                            if image_hash_1 in identical_hashes:
                                identical_hashes[image_hash_1].add(img_name_1)
                                identical_hashes[image_hash_1].add(img_name_2)
                            else:
                                identical_hashes[image_hash_1] = {img_name_1, img_name_2}
                # if the hashes are close, add to high similarity list
                elif image_hash_1 - image_hash_2 < cutoff:
                    # loop through high similarity list and check if one of the image names is already in there
                    added_to_existing_set = False
                    for img_set in high_similarity:
                        if img_name_1 in img_set or img_name_2 in img_set:
                            img_set.add(img_name_1)
                            img_set.add(img_name_2)
                            added_to_existing_set = True
                    # if not, add a new set with both image names
                    if not added_to_existing_set:
                        high_similarity.append({img_name_1, img_name_2})

    # convert dicts to lists
    identical_bytes = [list(img_names) for img_names in identical_bytes.values()]
    identical_hashes = [list(img_names) for img_names in identical_hashes.values()]
    return identical_bytes, identical_hashes, high_similarity



@click.command()
@click.option("--in_path_dfuc", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--in_path_fuseg", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
def convert_cfu(in_path_dfuc: str, in_path_fuseg: str, out_path: str):
    """
    Conversion script for the Combined Foot Ulcers (CFU) dataset. The dataset is simply a combination of the DFUC and
    FUSeg datasets. The dataset is filtered for duplicates and similar images.

    Args:
    in_path_dfuc (str): Path to the original dfuc data directory.
    in_path_fuseg (str): Path to the original fuseg data directory.
    out_path (str): Path where the converted data should be saved.

    Returns:
    None
    """
    # get dfuc paths
    dfuc_train_path_in = os.path.join(in_path_dfuc, "train")
    dfuc_train_img_path = os.path.join(dfuc_train_path_in, "DFUC2022_train_images")
    dfuc_train_mask_path = os.path.join(dfuc_train_path_in, "DFUC2022_train_masks")

    # get fuseg paths
    fuseg_train_path_in = os.path.join(in_path_fuseg, "train")
    fuseg_val_path_in = os.path.join(in_path_fuseg, "validation")
    fuseg_train_img_path = os.path.join(fuseg_train_path_in, "images")
    fuseg_train_mask_path = os.path.join(fuseg_train_path_in, "labels")
    fuseg_val_img_path = os.path.join(fuseg_val_path_in, "images")
    fuseg_val_mask_path = os.path.join(fuseg_val_path_in, "labels")

    check_in_folder_paths(dfuc_train_path_in, dfuc_train_img_path, dfuc_train_mask_path, fuseg_train_path_in,
                          fuseg_val_path_in, fuseg_train_img_path, fuseg_train_mask_path, fuseg_val_img_path,
                          fuseg_val_mask_path)


    img_paths_in = [dfuc_train_img_path, fuseg_train_img_path, fuseg_val_img_path]
    mask_paths_in = [dfuc_train_mask_path, fuseg_train_mask_path, fuseg_val_mask_path]

    # find similar images and create blacklist
    identical_bytes, identical_hashes, high_similarity = _find_similar(img_paths_in, cutoff=11)
    merged_lists = identical_bytes + identical_hashes + high_similarity
    merged_lists = [sorted(list(x)) for x in merged_lists]
    exclude_files = set()

    for file_list in merged_lists:
        if len(file_list) > 1:
            # remove last file from the list, because we want to keep one of the identical / similar files
            file_list.pop(-1)
        exclude_files.update(file_list)

    class_dict = {"background": 0, "wound": 255}
    folders_in = list(zip(img_paths_in, mask_paths_in))
    ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2}
    create_dataset(folders_in, out_path, class_dict, ratios, exclude_files=exclude_files)


if __name__ == "__main__":
    convert_cfu()

