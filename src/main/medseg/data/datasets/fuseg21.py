from typing import List, Optional

from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType

FUSEG21_DEFAULT_PATH = "{project_root}/data/datasets/fuseg21"


class FUSeg21(MedsegDataset):

    def __init__(
            self,
            split_type: SplitType,
            dataset_path: str = FUSEG21_DEFAULT_PATH,
            use_custom_split: bool = False,
            custom_split_ratios: Optional[List[float]] = None,
            randomize_custom_split: bool = False,
            random_seed: int = 42,
            indices: Optional[List[int]] = None
    ):
        super().__init__(split_type, dataset_path, use_custom_split=use_custom_split,
                         custom_split_ratios=custom_split_ratios, randomize_custom_split=randomize_custom_split,
                         random_seed=random_seed, indices=indices)
