from typing import Iterator

from torch import optim
from torch.nn.parameter import Parameter

OPTIMIZER_MAPPINGS = {
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamax": optim.Adamax,
    "adamw": optim.AdamW,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "sgd": optim.SGD,
    "sparseadam": optim.SparseAdam,
}


def get_optimizer(cfg: dict, param_iter: Iterator[Parameter]) -> optim.Optimizer:
    optim_cfg = cfg["optimizer"]
    optimizer = OPTIMIZER_MAPPINGS[optim_cfg["type"].lower()]
    # get args from optim_cfg
    args = {k: v for k, v in optim_cfg.items()}
    # remove type as it will not be a valid argument for the optimizer constructor
    args.pop("type")
    return optimizer(param_iter, **args)
