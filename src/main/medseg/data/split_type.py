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
