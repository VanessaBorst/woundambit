import os

import click

from medseg.data.converters.converters import create_dataset
from medseg.data.converters.helpers import check_in_folder_paths
from medseg.tools.datasets.check_for_duplicates import find_similar


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
    identical_bytes, identical_hashes, high_similarity = find_similar(img_paths_in, cutoff=11)
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

