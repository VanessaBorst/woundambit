from typing import Union

from torch import nn

from medseg.training.loss.caranet_loss import CaraNetLoss
from medseg.training.loss.combinations import BCEDice  # , BDiceFocal, BCEFocal, BCEFocalDice
from medseg.training.loss.focal_loss import BinaryFocalTverskyLogitsLoss
from medseg.training.loss.hardnet_dfus_loss import HarDNetDFUSLoss
from medseg.training.loss.soft_dice_loss import SoftDiceLoss
from medseg.util.class_ops import get_class_arg_names

LOSS_MAPPINGS = {
    "crossentropy": nn.CrossEntropyLoss,
    "softdice": SoftDiceLoss,
    "binaryfocaltverskylogits": BinaryFocalTverskyLogitsLoss,
    "bfocallogits": BinaryFocalTverskyLogitsLoss,
    "bftl": BinaryFocalTverskyLogitsLoss,
    "bce": nn.BCELoss,
    "binarycrossentropy": nn.BCELoss,
    "bcewithlogits": nn.BCEWithLogitsLoss,
    "binarycrossentropywithlogits": nn.BCEWithLogitsLoss,
    "bcelogits": nn.BCEWithLogitsLoss,
    "bcel": nn.BCEWithLogitsLoss,
    "bcewl": nn.BCEWithLogitsLoss,
    "bcedice": BCEDice,
    'ce': nn.CrossEntropyLoss,
    # "bdicefocal": BDiceFocal,
    # "bcefocal": BCEFocal,
    # "bcefocaldice": BCEFocalDice,
    "hardnetdfus": HarDNetDFUSLoss,
    "caranet": CaraNetLoss,
}


def get_loss_module(cfg: dict) -> Union[nn.Module, None]:
    """
     Create a loss module based on the configuration provided.

     Args:
         cfg (Dict[str, Any]): The configuration dictionary containing the loss type and its parameters.

     Returns:
         Union[nn.Module, None]: The instantiated loss module or None if not found in the configuration.
     """
    if "loss" not in cfg:
        return None
    loss_cfg = cfg["loss"].copy()
    loss_type = loss_cfg.pop("type")
    loss_type_parsed = loss_type.lower().replace(" ", "").replace("_", "").replace("loss", "")
    if loss_type_parsed == 'default':
        return None  # return None to trigger fetching the default loss function from the model
    if loss_type_parsed not in LOSS_MAPPINGS:
        raise ValueError(f"Loss type {loss_type_parsed} not supported")
    init_arg_names = get_class_arg_names(LOSS_MAPPINGS[loss_type_parsed])
    loss_args = {}
    for k, v in loss_cfg.items():
        if k in init_arg_names:
            loss_args[k] = v
        else:
            print(f"Warning: Loss {loss_type} does not support argument {k} defined in the config. Ignoring it.")
    return LOSS_MAPPINGS[loss_type_parsed](**loss_args)
