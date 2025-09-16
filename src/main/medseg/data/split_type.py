# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from enum import Enum


class SplitType(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    ALL = 'all'

    def get_full_name(self):
        if self == SplitType.TRAIN:
            return "Training"
        elif self == SplitType.VAL:
            return "Validation"
        elif self == SplitType.TEST:
            return "Test"
        elif self == SplitType.ALL:
            return "All"
