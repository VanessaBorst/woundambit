from typing import Tuple, List, Any, Optional

import torch
from torch.utils.data import DataLoader, default_collate

from medseg.data import datasets
from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType
from medseg.util.class_ops import get_class
from medseg.util.random import seed_worker


class DatasetManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.num_workers = cfg['settings'].get('num_workers', 0) if 'settings' in cfg else 0
        self.batch_size = cfg['settings'].get('batch_size', 1) if 'settings' in cfg else 1
        self.pin_memory = cfg['settings'].get('pin_memory', True) if 'settings' in cfg else True
        self.ds_train = self.build_dataset(self.cfg, SplitType.TRAIN)
        self.ds_val = self.build_dataset(self.cfg, SplitType.VAL)
        self.ds_test = self.build_dataset(self.cfg, SplitType.TEST)
        self.datasets = {
            SplitType.TRAIN: self.ds_train,
            SplitType.VAL: self.ds_val,
            SplitType.TEST: self.ds_test
        }

        self.aux_test_datasets_and_loaders = []
        aux_test_datasets = self.build_aux_test_datasets()
        for aux_test_dataset in aux_test_datasets:
            self.aux_test_datasets_and_loaders.append((aux_test_dataset, self.build_loader(aux_test_dataset)))

        self.loader_train = self.build_loader(self.ds_train) if self.ds_train is not None else None
        self.loader_val = self.build_loader(self.ds_val) if self.ds_val is not None else None
        self.loader_test = self.build_loader(self.ds_test) if self.ds_test is not None else None

        self.loaders = {
            SplitType.TRAIN: self.loader_train,
            SplitType.VAL: self.loader_val,
            SplitType.TEST: self.loader_test
        }

    @staticmethod
    def build_dataset(cfg: dict, split_type: SplitType, include_transforms_manager=False) -> Optional[MedsegDataset]:
        """
        Build a MedsegDataset based on the provided configuration and split type.

        Args:
            cfg (dict): The configuration dictionary.
            split_type (SplitType): The split type (TRAIN, VAL, or TEST).
            indices (Optional[List[int]]): Custom indices to use for the dataset (defines which samples are in
            the dataset split).

        Returns:
            MedsegDataset: A dataset instance for the given split type.
        """
        dataset_type = cfg["dataset"]["type"]
        dataset_class = get_class(datasets, dataset_type)
        assert dataset_class is not None, f"Could not map dataset class {dataset_type}"
        is_class_type_valid = isinstance(dataset_class, MedsegDataset) or issubclass(dataset_class, MedsegDataset)
        assert is_class_type_valid
        ds = dataset_class.from_cfg(cfg, split_type)
        # If the dataset does not have any samples for the specific split type, return None
        if len(ds) == 0:
            return None
        if include_transforms_manager:
            ds.set_transforms_manager_from_cfg(cfg)
        return ds

    def build_aux_test_datasets(self):
        aux_test_dataset_types = self.cfg['dataset'].get('aux_test_datasets', [])
        aux_test_datasets = []
        for dataset_type in aux_test_dataset_types:
            dataset_class = get_class(datasets, dataset_type)
            assert dataset_class is not None, f"Could not map dataset class {dataset_type}"
            is_class_type_valid = isinstance(dataset_class, MedsegDataset) or issubclass(dataset_class, MedsegDataset)
            assert is_class_type_valid
            ds = dataset_class(SplitType.TEST, use_custom_split=True, custom_split_ratios=[0, 0, 1.0])
            ds.set_transforms_manager_from_cfg(self.cfg)
            aux_test_datasets.append(ds)
        return aux_test_datasets

    def build_loader(self, dataset: MedsegDataset) -> DataLoader:
        """
        Build a data loader for the given dataset.

        Args:
            dataset (MedsegDataset): Dataset to build the loader for.

        Returns:
            DataLoader: DataLoader for the given dataset.
        """

        g = torch.Generator()
        seed = self.cfg['settings'].get('random_seed', 42) if 'settings' in self.cfg else 42
        g.manual_seed(seed)

        shuffle = dataset.split_type == SplitType.TRAIN
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=id_collate,
            worker_init_fn=seed_worker,
            generator=g
        )
        return loader

    def has_split(self, split: SplitType):
        return split in self.datasets and self.datasets[split] is not None

    def has_test_or_aux_test_split(self):
        return self.has_split(SplitType.TEST) or self.has_aux_test_datasets()

    def has_aux_test_datasets(self):
        return len(self.aux_test_datasets_and_loaders) > 0

    def get_dataset_and_loader(self, split: SplitType) -> Tuple[MedsegDataset, DataLoader]:
        return self.datasets[split], self.loaders[split]

    def get_aux_test_datasets_and_loaders(self) -> List[Tuple[MedsegDataset, DataLoader]]:
        return self.aux_test_datasets_and_loaders

    def get_dataset(self, split: SplitType):
        return self.datasets[split]

    def get_train_dataset(self) -> Optional[MedsegDataset]:
        return self.ds_train

    def get_val_dataset(self) -> Optional[MedsegDataset]:
        return self.ds_val

    def get_test_dataset(self) -> Optional[MedsegDataset]:
        return self.ds_test

    def get_loader(self, split: SplitType):
        return self.loaders[split]

    def get_train_loader(self):
        return self.loader_train

    def get_val_loader(self):
        return self.loader_val

    def get_test_loader(self):
        return self.loader_test


def id_collate(batch: List[Any]) -> Tuple[Any, List[Any]]:
    """
      Collate function that returns the ids of the batch items, in addition to the image and label tensors. This
      allows for the assignment of metrics to each image and identification of images where the model performed poorly.
      Adapted from https://discuss.pytorch.org/t/building-custom-dataset-how-to-return-ids-as-well/22931/7

      Args:
          batch (list): A list of batch items.

      Returns:
          tuple: A tuple containing the collated batch data and the corresponding ids.
      """
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids
