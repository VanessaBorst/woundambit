import os
from abc import ABCMeta
from typing import List, Tuple, Optional, Union, Dict

import numpy as np
import pandas as pd
from PIL.Image import Image
from beartype import beartype
from torch.utils import data

from medseg.data.class_mapping import ClassMapping
from medseg.data.split_type import SplitType
from medseg.data.transforms.transforms_manager import TransformsManager
from medseg.util.img_ops import open_image, get_pil_mode
from medseg.util.path_builder import resolve_placeholder_path

from medseg.util.random import get_splits

DEFAULT_IMG_SUBDIR = "images"
DEFAULT_MASK_SUBDIR = "masks"
DEFAULT_DATASET_PATH = "{project_root}/data/datasets/ds"


class MedsegDataset(data.Dataset, metaclass=ABCMeta):
    """
    Abstract base class for image segmentation datasets.
    """

    @beartype
    def __init__(
            self,
            split_type: SplitType,
            dataset_path: Optional[str] = None,
            use_custom_split: bool = False,
            custom_split_ratios: Optional[List[Union[float, int]]] = None,
            randomize_custom_split: bool = False,
            random_seed: int = 42,
            indices: Optional[List[int]] = None,
            img_subdir=DEFAULT_IMG_SUBDIR,
            mask_subdir=DEFAULT_MASK_SUBDIR
    ):
        self.transforms = {}
        self.split_type = split_type
        self.dataset_path = resolve_placeholder_path(dataset_path)
        self.img_channels = 3
        self.mask_channels = 1
        self.img_mode = get_pil_mode(self.img_channels)
        self.mask_mode = get_pil_mode(self.mask_channels)
        self.index_df = self.read_dataset_index()

        self.class_mapping = ClassMapping(self.get_class_defs())

        self.is_default_split = True  # flag might change in the code below
        self.train_split_mean_std = None

        self.use_custom_split = use_custom_split
        self.custom_split_ratios = custom_split_ratios
        self.randomize_custom_split = randomize_custom_split
        self.random_seed = random_seed

        if indices is None:
            self.indices = self.determine_indices()
        else:
            if self.use_custom_split:
                print("Warning: custom_split is set to True, but indices for the dataset were provided. "
                      "Ignoring custom_split.")
            assert (isinstance(indices, list) and len(indices) > 0)
            self.indices = indices
        self.n = len(self.indices)

        self.all_images = self.index_df["image"].tolist()
        self.all_masks = self.index_df["mask"].tolist()
        self.images = [self.all_images[i] for i in self.indices]
        self.masks = [self.all_masks[i] for i in self.indices]
        assert len(self.images) == len(self.masks) == len(self.indices)
        self.image_paths = [os.path.join(self.dataset_path, img_subdir, fname) for fname in self.images]
        self.mask_paths = [os.path.join(self.dataset_path, mask_subdir, fname) for fname in self.masks]
        assert len(self.image_paths) == len(self.mask_paths)

        self.uniform_image_dimensions = self.index_df["height"].nunique() == 1 and self.index_df["width"].nunique() == 1
        if self.n > 0:
            self.default_hw = self.get_default_height_width()
        self.transforms_manager = None

    @classmethod
    @beartype
    def from_cfg(cls, cfg: dict, split_type: SplitType, disable_transforms_manager=False):
        random_seed = cfg["dataset"].get("random_seed", 42)
        use_custom_split = cfg["dataset"].get("use_custom_split", False)
        custom_split_ratios = cfg["dataset"].get("custom_split_ratios", None)
        randomize_custom_split = cfg["dataset"].get("randomize_custom_split", False)
        indices = cfg["dataset"].get(f"{split_type.value.lower()}_indices", None)
        ds = cls(split_type, use_custom_split=use_custom_split, custom_split_ratios=custom_split_ratios,
                 randomize_custom_split=randomize_custom_split, random_seed=random_seed, indices=indices)
        if ds.n > 0 and not disable_transforms_manager and split_type != SplitType.ALL:
            transforms_manager = TransformsManager(cfg, split_type,
                                                   default_hw=ds.default_hw, get_height_width=ds.get_height_width)
            ds.set_transforms_manager(transforms_manager)
        return ds

    def determine_indices(self) -> List[int]:
        indices = []
        if self.split_type == SplitType.ALL:
            indices = self.index_df.index.tolist()

        elif self.use_custom_split is True and self.custom_split_ratios is not None:
            if not isinstance(self.custom_split_ratios, list) and not isinstance(self.custom_split_ratios, tuple):
                raise ValueError(f"custom_split_ratios must be a list or tuple. Got {self.randomize_custom_split}")
            self.is_default_split = False
            n_all = self.index_df.shape[0]
            all_indices = np.array(list(range(n_all)))
            # set a random seed in if randomize_custom_split is True
            split_seed = self.random_seed if self.randomize_custom_split else None
            # get indices for the train, val, and test sets
            train_idx, val_idx, test_idx = get_splits(all_indices, self.custom_split_ratios[0],
                                                      self.custom_split_ratios[1], split_seed)
            if self.split_type == SplitType.TRAIN:
                indices = train_idx.tolist()
            elif self.split_type == SplitType.VAL:
                indices = val_idx.tolist()
            elif self.split_type == SplitType.TEST:
                indices = test_idx.tolist()

        else:
            # use the default split set in the index.csv file
            indices = self.index_df[self.index_df["split"] == self.split_type.value].index.tolist()

        return indices

    @beartype
    def get_name(self) -> str:
        name = self.__class__.__name__
        if name.lower() == "medsegdataset":
            name = self.dataset_path.split(os.sep)[-1]  # can return empty string if last char is os.sep
            if name == "":
                name = self.dataset_path.split(os.sep)[-2]
        return name

    @beartype
    def get_default_height_width(self) -> Optional[Tuple[int, int]]:
        default_hw = None
        if self.uniform_image_dimensions:
            first_index = self.dataset_index_to_real_index(0)
            default_hw = int(self.index_df.iloc[first_index]["height"]), int(self.index_df.iloc[first_index]["width"])
        return default_hw

    @beartype
    def set_transforms_manager_from_cfg(self, cfg: dict):
        transforms_manager = TransformsManager(cfg, self.split_type,
                                               default_hw=self.default_hw, get_height_width=self.get_height_width)
        self.set_transforms_manager(transforms_manager)

    @beartype
    def set_transforms_manager(self, transforms_manager: TransformsManager):
        self.transforms_manager = transforms_manager

    @beartype
    def __len__(self):
        return len(self.indices)

    @beartype
    def __getitem__(self, i: int):
        img, mask, real_i = self.load_img_mask(i)
        img, mask = self.transforms_manager.apply_transforms(img, mask, real_i)
        mask = self.class_mapping.apply_class_mapping(mask)
        return img, mask, real_i

    @beartype
    def load_img_mask(self, i: int) -> Tuple[Image, Image, int]:
        img_path = self.image_paths[i]
        mask_path = self.mask_paths[i]
        real_i = self.dataset_index_to_real_index(i)
        img = open_image(img_path, self.index_df.iloc[real_i]["img_type"]).convert(self.img_mode)
        mask = open_image(mask_path, self.index_df.iloc[real_i]["mask_type"]).convert(self.mask_mode)
        return img, mask, real_i

    @beartype
    def load_img(self, i: int) -> Image:
        img_path = self.image_paths[i]
        real_i = self.dataset_index_to_real_index(i)
        return open_image(img_path, self.index_df.iloc[real_i]["img_type"]).convert(self.img_mode)

    @beartype
    def load_mask(self, i: int) -> Image:
        mask_path = self.mask_paths[i]
        real_i = self.dataset_index_to_real_index(i)
        return open_image(mask_path, self.index_df.iloc[real_i]["mask_type"]).convert(self.mask_mode)

    @beartype
    def real_index_to_dataset_index(self, real_i: int) -> int:
        """
        Converts the actual index number corresponding to the entry in index.csv to the dataset index.
        :param real_i: the actual index number
        """
        return self.indices.index(real_i)

    @beartype
    def dataset_index_to_real_index(self, i: int):
        """
        Converts the dataset index to the actual index number corresponding to the entry in index.csv
        :param i: the index
        :return: the actual index number
        """
        return self.indices[i]

    @beartype
    def get_height_width(self, real_i: int) -> Tuple[int, int]:
        """
        Return the stored height and width of the image at the given index
        :param real_i: the index
        :return: tuple of height and width
        """
        return int(self.index_df.iloc[real_i]["height"]), int(self.index_df.iloc[real_i]["width"])

    @beartype
    def get_image_file_name(self, real_i: int) -> str:
        """
        Returns the image file name for the given index
        :param real_i: the index number
        :return: the image file name
        """
        return self.all_images[real_i]

    @beartype
    def read_dataset_index(self) -> pd.DataFrame:
        """
        Reads the index.csv file and returns it as a pandas DataFrame
        :return: the index DataFrame
        """
        index_path = os.path.join(self.dataset_path, "index.csv")
        df = pd.read_csv(index_path)
        return df

    @beartype
    def is_multiclass(self) -> bool:
        """
        Returns whether the dataset is multiclass or not
        :return: True if multiclass, False otherwise
        """
        return self.class_mapping.multiclass

    @property
    @beartype
    def background_class(self) -> dict:
        """
        Returns the background class dictionary
        """
        return self.class_mapping.background_class

    @beartype
    def get_class_defs(self) -> List[Dict[str, Union[str, int]]]:
        """
        Reads the classes.csv file and returns it as a list of dicts
        :return: the classes containing a "label" key and a "pixel_value" key for each class
        """
        index_path = os.path.join(self.dataset_path, "classes.csv")
        df = pd.read_csv(index_path, index_col=[0])
        return df.to_dict('records')

