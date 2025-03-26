import os
import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Tuple

import click
import imagehash
from beartype import beartype

from medseg.util.img_ops import open_image

SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}


@click.command()
@click.option("--path", type=str, required=True, multiple=True,
              help="Paths to the folders containing the images to be checked. "
                   "Separate multiple paths with spaces.")
@click.option("--cutoff", type=int, default=8, required=False,
              help="The cutoff value for considering two images similar based on their hash difference. "
                   "The lower the cutoff value, the higher the similarity required for detection. Default is 8.")
@click.option("--display_full_paths", is_flag=True, default=False, required=False)
def check_for_duplicates(path: Tuple[str], cutoff: int = 8, display_full_paths: bool = False) -> None:
    """Checks a folder containing images for duplicates by comparing their image hashes."""
    paths = list(path)
    identical_bytes, identical_hashes, high_similarity = find_similar(paths, cutoff,
                                                                      display_full_paths=display_full_paths)
    print(create_duplicates_text_summary(identical_bytes, identical_hashes, high_similarity))


@beartype
def create_duplicates_text_summary(identical_bytes: list, identical_hashes: list, high_similarity: list) -> str:
    """Creates a textual representation of the potential duplicate images found in the folders, grouped by different
    levels of similarity.

      Args:
          identical_bytes (list): A list of sets of image names that are identical in bytes.
          identical_hashes (list): A list of sets of image names that have identical image hashes, but not identical bytes.
          high_similarity (list): A list of sets of image names that have high similarity based on their image hashes.
      """

    output_lines = []
    output_lines.append("Definitive duplicates (identical bytes):")
    output_lines.append(f"Sets of byte-duplicates found: {len(identical_bytes)}")
    output_lines.append(f"Total number of byte-duplicate images found: {sum([len(x) for x in identical_bytes])}")
    output_lines.append(f"Number of byte-duplicate images to remove: {sum([len(x) - 1 for x in identical_bytes])}")
    if len(identical_bytes) == 0:
        output_lines.append("None found.")
    else:
        for i, img_names in enumerate(identical_bytes):
            output_lines.append(f"{i + 1}: {img_names}")
    output_lines.append("----------------------------------------------------------------------------------")
    output_lines.append("Identical image hash, but not identical bytes:")
    output_lines.append(f"Sets of hash-duplicates found: {len(identical_hashes)}")
    output_lines.append(f"Total number of hash-duplicate images found: {sum([len(x) for x in identical_hashes])}")
    output_lines.append(f"Number of hash-duplicate images to remove: {sum([len(x) - 1 for x in identical_hashes])}")
    if len(identical_hashes) == 0:
        output_lines.append("None found.")
    else:
        for i, img_names in enumerate(identical_hashes):
            output_lines.append(f"{i + 1}: {img_names}")
    output_lines.append("----------------------------------------------------------------------------------")
    output_lines.append(f"Sets of similar images found: {len(high_similarity)}")
    output_lines.append(f"Total number of similar images found: {sum([len(x) for x in high_similarity])}")
    output_lines.append(f"Number of similar images to remove: {sum([len(x) - 1 for x in high_similarity])}")
    output_lines.append("Possible similarity:")
    if len(high_similarity) == 0:
        output_lines.append("None found.")
    else:
        for i, img_names in enumerate(high_similarity):
            output_lines.append(f"{i + 1}: {img_names}")
    output_text = "\n".join(output_lines)
    return output_text


@beartype
def find_similar(paths: List[str], cutoff=11, display_full_paths: bool = False) -> Tuple[List, List, List]:
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


@beartype
def extract_similar(identical_bytes: list, identical_hashes: list, high_similarity: list, in_path: str,
                    out_path: str):
    """Copies each set of similar images to their own folder in the output path.

    Identical sets will be in a folder named identical_bytes_i, where i is the index of the set
    in the list of identical sets. Similarly, sets with identical hashes will be in a folder
    named identical_hashes_i, and sets with high similarity will be in a folder named
    high_similarity_i.

    Args:
        identical_bytes (list): A list of sets of images with identical bytes.
        identical_hashes (list): A list of sets of images with identical hashes.
        high_similarity (list): A list of sets of images with high similarity.
        in_path (str): The input path containing the original images.
        out_path (str): The output path where the similar images will be copied to.
    """

    Path(out_path).mkdir(parents=True, exist_ok=True)

    def copy_files(list_of_file_sets: list, in_path: str, out_path: str, out_folder_prefix: str):
        for i, file_name_sets in enumerate(list_of_file_sets):
            # create folder if it doesn't exist yet and copy images there
            folder_name = f"{out_folder_prefix}_{i}"
            to_path = os.path.join(out_path, folder_name)
            Path(to_path).mkdir(parents=True, exist_ok=True)
            for file_name in file_name_sets:
                shutil.copy(os.path.join(in_path, file_name), os.path.join(to_path, file_name))

    copy_files(identical_bytes, in_path, out_path, "identical_bytes")
    copy_files(identical_hashes, in_path, out_path, "identical_hashes")
    copy_files(high_similarity, in_path, out_path, "high_similarity")


if __name__ == "__main__":
    check_for_duplicates()
