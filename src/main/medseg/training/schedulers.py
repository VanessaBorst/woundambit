import inspect
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import *

from medseg.evaluation.metrics import EvalMetric


def get_scheduler(cfg: dict, optim: Optimizer, last_epoch: int = -1) \
        -> Union[ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, None]:
    """
    Get the learning rate scheduler instance based on the given configuration.

    Args:
        cfg (dict): A dictionary containing the scheduler configuration.
        optim (Optimizer): The optimizer instance for which the scheduler will be used.
        last_epoch (int, optional): The index of the last epoch. Default is -1.

    Returns:
        Union[ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, None]: An instance of the scheduler.
    """

    if 'scheduler' not in cfg \
            or cfg['scheduler'] is None \
            or cfg['scheduler'] is False \
            or 'type' not in cfg['scheduler'] \
            or not isinstance(cfg['scheduler']['type'], str) \
            or cfg['scheduler']['type'].lower() not in SCHEDULER_MAPPINGS:
        return None
    scheduler_cfg = cfg['scheduler']
    scheduler = SCHEDULER_MAPPINGS[scheduler_cfg['type'].lower()]
    # get args from scheduler_cfg
    args = {k: v for k, v in scheduler_cfg.items()}
    # add last_epoch if it is a valid argument for the scheduler constructor
    all_scheduler_args = inspect.getfullargspec(scheduler).args
    if 'last_epoch' in all_scheduler_args and 'last_epoch' not in args:
        args['last_epoch'] = last_epoch

    if 'T_max' in all_scheduler_args and 'T_max' not in args:
        args['T_max'] = cfg['settings']['max_epochs']

    # remove type and metric as it will not be a valid argument for the scheduler constructor
    if 'type' in args: del args['type']
    if 'metric' in args: del args['metric']
    return scheduler(optim, **args)


def build_scheduler_and_metric(optimizer, cfg: dict):
    """
    Build the learning rate scheduler and evaluation metric based on the given configuration.

    Args:
        optimizer (Optimizer): The optimizer instance for which the scheduler will be used.
        cfg (dict): A dictionary containing the scheduler configuration.

    Returns:
        tuple: A tuple containing the scheduler instance and evaluation metric instance.
    """
    scheduler = get_scheduler(cfg, optimizer)
    scheduler_metric = None
    if scheduler is not None and cfg['scheduler']['type'].lower() in SCHEDULERS_WITH_METRIC:
        assert 'metric' in cfg['scheduler'], \
            f"Metric must be specified in config for {cfg['scheduler']['type']} scheduler."
        scheduler_metric = EvalMetric(cfg['scheduler']['metric'])
    return scheduler, scheduler_metric


def get_rex_scheduler(optimizer, T_max: int) -> LambdaLR:
    rex = lambda t: (1 - t / T_max) / (0.5 + 0.5 * (1 - t / T_max))
    scheduler = LambdaLR(optimizer, lr_lambda=rex)
    return scheduler


SCHEDULER_MAPPINGS = {
    'reduce_lr_on_plateau': ReduceLROnPlateau,
    'cosine_annealing': CosineAnnealingLR,
    'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts,
    'polynomial': PolynomialLR,
    'rex': get_rex_scheduler
}

SCHEDULERS_WITH_METRIC = {
    'reduce_lr_on_plateau',
}
