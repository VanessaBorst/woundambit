# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


from collections import OrderedDict
from typing import List, Dict, Union
from typing import Optional

import torch
from beartype import beartype
from torch import Tensor


class ClassMapping:
    """
    Class to map pixel values to segmentation labels and class indices.
    """

    @beartype
    def __init__(self, class_defs: List[Dict[str, Union[str, int]]], default_pixel_value: Optional[int]=0):
        """
        Initializes the ClassMapping object with class definitions.

        This constructor requires a list of dictionaries, where each dictionary represents a class definition. Each class
        definition must include a 'label' and a 'pixel_value'. The 'label' is a string identifier for the class, and
        the 'pixel_value' is an integer representing the pixel value corresponding to that class in segmentation masks.

        It is mandatory to include a background class in the class definitions with the label 'background'. This class
        is used to represent the background in segmentation tasks and is typically mapped to pixel value 0. However,
        if a different value is provided in the definitions, it will be used instead.

        The method sets up two primary mappings:
        1. Indices to pixels mapping: Used for translating class indices to their corresponding pixel values.
        2. Pixels to indices mapping: Used for translating pixel values to their corresponding class indices.

        These mappings are crucial for converting between raw pixel values in segmentation masks and the semantic
        classes they represent.

        Parameters:
        -----------
        class_defs : List[Dict[str, Union[str, int]]]
            A list of dictionaries, each representing a class definition. Each dictionary must have two keys:
            - 'label' (str): The name of the class.
            - 'pixel_value' (int): The pixel value assigned to this class in segmentation masks.

        Raises:
        -------
        ValueError:
            If any class other than the background class shares the same pixel value as the background class.
            If the 'pixel_value' provided is not an integer.

        Examples:
        ---------
        >>> class_definitions = [
                {"label": "background", "pixel_value": 0},
                {"label": "class1", "pixel_value": 1},
                {"label": "class2", "pixel_value": 2},
            ]
        >>> class_mapping = ClassMapping(class_definitions)

        Notes:
        ------
        - The 'background' class is a special class used to represent areas in an image that do not belong to any
          of the other classes. It is essential for segmentation tasks.
        - The class indices are implicitly assigned based on the order of the class definitions in the 'class_defs' list,
          with the background class always assigned index 0.
        """
        assert len(class_defs) > 0
        self.bg_class_def = next((c for c in class_defs if c["label"].lower() == "background"), None)
        if self.bg_class_def is None:
            raise Warning(
                f"Background class not found in class definitions. Assuming default value of 0. class with"
                f" the label 'background should be defined in the classes.csv of the respective dataset.")
        self.class_defs = class_defs
        self.default_pixel_value = default_pixel_value

        self._indices_to_pixels_mapping = self._create_indices_to_pixels_mapping()
        self._indices_to_pixels_tensor = torch.full((256,), self.default_pixel_value)
        for from_val, to_val in self._indices_to_pixels_mapping.items():
            self._indices_to_pixels_tensor[from_val] = to_val

        self._pixels_to_indices_mapping = self._create_pixels_to_indices_mapping()
        self._pixels_to_indices_tensor = torch.full((256,), self.default_pixel_value) #TODO

        for from_val, to_val in self._pixels_to_indices_mapping.items():
            self._pixels_to_indices_tensor[from_val] = to_val
        #attach pixels to the nearest value
        for pixel_val in range(256):
            if pixel_val not in self._pixels_to_indices_mapping:
                # Finde den nächsten bekannten Pixelwert
                nearest_val = min(self._pixels_to_indices_mapping.keys(),
                                  key=lambda x: abs(x - pixel_val))
                self._pixels_to_indices_tensor[pixel_val] = self._pixels_to_indices_mapping[nearest_val]
    @beartype
    def _create_indices_to_pixels_mapping(self) -> OrderedDict[int, int]:
        """
         Returns an ordered dictionary, mapping the class indices to pixels. The background pixel value is always
         mapped to 0 and can be found in the first position of the dictionary.
         :return: the class mappings
         """
        indices_to_pixels = OrderedDict()
        indices_to_pixels[0] = self.background_class["pixel_value"]
        i = 1
        for c in self.class_defs:
            if c["label"] == self.bg_class_def["label"]: continue
            assert c["pixel_value"] != self.bg_class_def["pixel_value"]
            assert isinstance(c["pixel_value"], int) and c["pixel_value"] != self.background_class["pixel_value"]
            indices_to_pixels[i] = c["pixel_value"]
            i += 1
        return indices_to_pixels

    @beartype
    def _create_pixels_to_indices_mapping(self) -> OrderedDict[int, int]:
        """
           Returns an ordered dictionary, mapping the pixel values to class indices. The background pixel value is always
           mapped to 0 and can be found in the first position of the dictionary.
           :return: the class mappings
       """

        indices_to_pixels = self._indices_to_pixels_mapping
        pixels_to_indices = OrderedDict()
        for k, v in indices_to_pixels.items():
            pixels_to_indices[v] = k
        return pixels_to_indices

    @property
    @beartype
    def background_class(self) -> Dict[str, Union[str, int]]:
        return self.bg_class_def

    @beartype
    def class_label_to_index(self, class_label: str) -> Optional[int]:
        pixel_val = next((c["pixel_value"] for c in self.class_defs if c["label"] == class_label), None)
        return self.pixel_to_index(pixel_val) if pixel_val is not None else None

    @property
    @beartype
    def num_classes(self) -> int:
        return len(self.class_defs)

    @property
    @beartype
    def pixels_to_indices_mapping(self) -> OrderedDict[int, int]:
        return self._pixels_to_indices_mapping

    @beartype
    @beartype
    def pixel_to_index(self, pixel: int) -> Optional[int]:
        return self._pixels_to_indices_mapping[pixel] if pixel in self._pixels_to_indices_mapping else None

    @property
    @beartype
    def indices_to_pixels_mapping(self) -> OrderedDict[int, int]:
        return self._indices_to_pixels_mapping

    @beartype
    def index_to_pixel(self, index: int) -> Optional[int]:
        return self._indices_to_pixels_mapping[index] if index in self._indices_to_pixels_mapping else None

    @property
    @beartype
    def multiclass(self) -> bool:
        return self.num_classes > 2

    @property
    @beartype
    def bg_index(self) -> int:
        return 0

    @property
    @beartype
    def bg_pixel(self) -> int:
        return self.background_class["pixel_value"]

    @beartype
    def apply_class_mapping(self, mask: Tensor) -> Tensor:
        """
        Applies the class mappings to the mask
        :param mask:  to apply the mappings to
        :return: the mask with the mappings applied
        """
        return self._pixels_to_indices_tensor[mask.long()]

    @beartype
    def revert_class_mapping(self, mask: Tensor) -> Tensor:
        """
        Reverts the class mappings to the mask
        :param mask:  to revert the mappings to
        :return: the mask with the mappings reverted
        """
        return self._indices_to_pixels_tensor[mask.long()]

    @beartype
    def state_dict(self) -> dict:
        return {
            "class_defs": self.class_defs,
        }

    @classmethod
    @beartype
    def from_dict(cls, state_dict: dict):
        return cls(state_dict["class_defs"])

