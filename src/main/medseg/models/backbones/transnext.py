# --------------------------------------------------------
# Code adapted by Vanessa Borst, Timo Dittus and Contributors
# from https://github.com/DaiShiResearch/TransNeXt/
#
# Original license from the TransNeXt repository:
# Apache License 2.0
# Copyright 2023 - present, Dai Shi
# --------------------------------------------------------


import torch
import torch.nn as nn
import pkg_resources


# Code based on https://github.com/DaiShiResearch/TransNeXt/blob/main/segmentation/upernet/train.py

def is_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


class TransNextModel(nn.Module):
    def __init__(self, variant="tiny", in_channels=3, img_size=512, is_extrapolation=None):
        super().__init__()

        # Determine the implementation based on the presence of 'swattention' package
        if is_installed('swattention') and torch.cuda.is_available():
            print('swattention package found, loading CUDA version of TransNeXt')
            import medseg.models.backbones.transnext_versions.transnext_cuda as transnext_impl
        else:
            print('swattention package not found, loading PyTorch native version of TransNeXt')
            import medseg.models.backbones.transnext_versions.transnext_native as transnext_impl

        # Store the chosen implementation
        self.transnext_impl = transnext_impl

        # Initialize the selected model variant
        self.model = self._initialize_model(variant, in_channels, img_size, is_extrapolation)

    def _initialize_model(self, variant, in_channels, img_size, is_extrapolation):
        # Create a model based on the selected variant
        if variant == "tiny":
            return self.transnext_impl.TransNeXtTiny(in_chans=in_channels, img_size=img_size,
                                                     is_extrapolation=is_extrapolation)
        elif variant == "small":
            return self.transnext_impl.TransNeXtSmall(in_chans=in_channels, img_size=img_size,
                                                      is_extrapolation=is_extrapolation)
        elif variant == "base":
            return self.transnext_impl.TransNeXtBase(in_chans=in_channels, img_size=img_size,
                                                     is_extrapolation=is_extrapolation)
        else:
            raise ValueError(f"Unknown TransNeXt variant: {variant}")

    def forward(self, x):
        """
        Calls the forward method of the selected TransNeXt model.
        """
        return self.model(x)
