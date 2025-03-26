import os
from collections import defaultdict

import click

from medseg.data.converters.converters import create_dataset
from medseg.data.converters.helpers import get_image_dims


@click.command()
@click.option("--in_path", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--out_path", type=click.Path(exists=False), required=True)
@click.option("--copy_files", type=bool, default=True, required=False)
def convert_ukw(in_path=None, out_path=None, copy_files=True):
    """
       Convert the original UKW dataset folder structure and add an index.csv and classes.csv file.
       Original structure is roughly as follows:
           - ukw (main directory)
              file_name_1.jpg
              file_name_1_mask_cb.JPG
              file_name_1_mask_sh.jpg
              file_name_2.jpg
              file_name_2_mask_cb.jpg
              file_name_3.jpg
              file_name_1_mask_cb.JPG
              file_name_1_mask_gs.jpg

       Out folder structure should be:
            - images (directory)
              file_name_1.jpg
              file_name_2.jpg
              file_name_3.jpg

            - masks (directory)
              file_name_1_mask_cb.JPG
              file_name_1_mask_sh.jpg
              file_name_2_mask_cb.jpg
              file_name_1_mask_cb.JPG
              file_name_1_mask_gs.jpg

        Args:
        in_path (str): Path to the original main directory.
        out_path (str): Path where the converted data should be saved.
        copy_files (bool): If True, the files will be copied to the out_path, otherwise only an index.csv, classes.csv,
        and possibly a blacklist.csv file will be created.

        Returns:
        None
       """

    file_names = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
    file_names_lower = [f.lower() for f in file_names]

    names_lower_to_original = {}
    for i, fname_lower in enumerate(file_names_lower):
        names_lower_to_original[fname_lower] = file_names[i]

    img_mask_dict = defaultdict(list)
    missing_imgs_derived_from_masks = set()

    for fname_lower, fname in names_lower_to_original.items():
        if "_mask" in fname_lower:
            base_name, ext = os.path.splitext(fname)
            img_name_l = (base_name.split("_mask")[0] + ext).lower()
            if img_name_l not in names_lower_to_original.keys():
                missing_imgs_derived_from_masks.add(img_name_l)
            else:
                img_name = names_lower_to_original[img_name_l]
                img_mask_dict[img_name].append(fname)
        else:
            if fname not in img_mask_dict.keys():
                img_mask_dict[fname] = []

    keys_to_remove = []
    for img_name, mask_names in img_mask_dict.items():
        # check image dimensions against mask dimensions, find mismatches
        if len(mask_names) > 0:
            img_path = os.path.join(in_path, img_name)
            img_h, img_w = get_image_dims(img_path, 'jpg')
            for mask_name in mask_names:
                mask_path = os.path.join(in_path, mask_name)
                mask_h, mask_w = get_image_dims(mask_path, 'jpg')
                if img_h != mask_h or img_w != mask_w:
                    keys_to_remove.append(img_name)

    for key in keys_to_remove:
        print(f"Skipping image {key}, because resolutions don't match between image and mask.")
        del img_mask_dict[key]

    img_paths_in = []
    mask_paths_in = []

    for img_name, mask_names in img_mask_dict.items():
        if len(mask_names) == 1:
            img_paths_in.append(os.path.join(in_path, img_name))
            mask_paths_in.append(os.path.join(in_path, mask_names[0]))
        else:
            print(f"Skipping image {img_name}, because it has {len(mask_names)} masks.")

    class_dict = {"background": 0, "wound": 255}
    paths_in = list(zip(img_paths_in, mask_paths_in))
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    create_dataset(paths_in, out_path, class_dict, ratios, strict_filename_matching=False, copy_files=copy_files)


if __name__ == "__main__":
    convert_ukw()
