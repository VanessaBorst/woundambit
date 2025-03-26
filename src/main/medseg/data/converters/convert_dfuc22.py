import os

import click

from medseg.data.converters.converters import create_dataset
from medseg.data.converters.helpers import check_in_folder_paths


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=True)
def convert_dfuc22(in_path=None, out_path=None, copy_files=True):
    """
    Convert the original DFUC22 folder structure and add an index.csv and classes.csv file.
    Original structure is as follows (please rename the folder from "DFUC2022_train_release" to "train"
    - DFU2022
        - train (2000 files in each sub folder)
            - DFUC2022_train_images: 100001.jpg, 100002.jpg, ..., 102000.jpg
            - DFUC2022_train_masks: 100001.jpg, 100002.jpg, ..., 102000.jpg
        - test (2000 files )
            - images: 200001.jpg, 200002.jpg, ..., 202000.jpg

    Args:
    in_path (str): Path to the original main directory.
    out_path (str): Path where the converted data should be saved.
    copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
    and possibly a blacklist.csv file will be created.

    Returns:
    None
    """

    train_path_in = os.path.join(in_path, "train")

    # get image and mask paths for train
    train_img_path = os.path.join(train_path_in, "DFUC2022_train_images")
    train_mask_path = os.path.join(train_path_in, "DFUC2022_train_masks")
    check_in_folder_paths(train_img_path, train_mask_path)
    folders_in = [(train_img_path, train_mask_path)]

    class_dict = {"background": 0, "wound": 255}
    ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2}
    create_dataset(folders_in, out_path, class_dict, ratios, copy_files=copy_files)


if __name__ == "__main__":
    convert_dfuc22()
