# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os

import click

from medseg.data.converters.converters import create_dataset
from medseg.data.converters.helpers import check_in_folder_paths


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=True)
def convert_fuseg21(in_path=None, out_path=None, copy_files=True):
    """
    Convert the original FUSeg21 folder structure and add an index.csv and classes.csv file.
    Original structure is as follows:
    - FUSeg21
        - train (some numbers missing, 810 files in each sub folder)
            - images: 0011.png, 0002.png, ..., 1010.png
            - labels: 0011.png, 0002.png, ..., 1010.png
        - test
            - images: 1011.png, 0002.png, ..., 1476.png
            - labels: <no labels>
        - validation (some numbers missing, 200 files in each sub folder)
             - images: 0001.png, 0002.png, ..., 1009.png
             - <no labels folder>


    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved.
    copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
    and possibly a blacklist.csv file will be created.

    Returns:
    None
    """

    train_path_in = os.path.join(in_path, "train")
    val_path_in = os.path.join(in_path, "validation")
    train_img_path = os.path.join(train_path_in, "images")
    train_mask_path = os.path.join(train_path_in, "labels")
    val_img_path = os.path.join(val_path_in, "images")
    val_mask_path = os.path.join(val_path_in, "labels")

    check_in_folder_paths(train_img_path, train_mask_path, val_img_path, val_mask_path)

    train_paths = (train_img_path, train_mask_path)
    val_paths = (val_img_path, val_mask_path)

    class_dict = {"background": 0, "wound": 255}
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

    create_dataset([train_paths, val_paths], out_path, class_dict, ratios, copy_files=copy_files)



if __name__ == "__main__":
    convert_fuseg21()
