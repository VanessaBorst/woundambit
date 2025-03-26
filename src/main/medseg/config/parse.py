import re
from typing import Union

from beartype import beartype
from hyperopt import hp
from torchvision.transforms import InterpolationMode

from medseg.evaluation.metrics import EvalMetric, MetricsMethod

INTERPOLATION_MAPPINGS = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'box': InterpolationMode.BOX,
    'hamming': InterpolationMode.HAMMING,
    'lanczos': InterpolationMode.LANCZOS
}


@beartype
def parse_all_tuples(cfg: Union[dict, list]) -> Union[dict, list]:
    """
    Parses all tuples in the config from strings to tuples.

    Args:
        cfg (Union[dict, list]): The config dictionary or list.

    Returns:
        Union[dict, list]: The parsed config dictionary or list.
    """

    def recursion_if_list_dict(val: any) -> any:
        return parse_all_tuples(val) if isinstance(val, dict) or isinstance(val, list) else val

    if isinstance(cfg, list):
        for i in range(len(cfg)):
            cfg[i] = parse_tuple(cfg[i]) if isinstance(cfg[i], str) else recursion_if_list_dict(cfg[i])
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            cfg[key] = parse_tuple(value) if isinstance(value, str) else recursion_if_list_dict(value)
    return cfg


@beartype
def parse_tuple(tuple_string: str) -> Union[str, tuple]:
    """
    Parses a string of a tuple to a tuple (if it is a valid tuple string).

    Args:
        tuple_string (str): The string to parse.

    Returns:
        tuple: The parsed tuple.
    """

    if tuple_string.startswith("(") and tuple_string.endswith(")") and re.search('[a-zA-Z]', tuple_string) is None:
        def convert_to_number(x):
            try:
                return int(x.strip())
            except ValueError:
                return float(x.strip())

        tuple_string = tuple([convert_to_number(x) for x in tuple_string[1:-1].split(",")])

    return tuple_string


@beartype
def parse_interpolation(cfg: dict) -> dict:
    """
    Replaces config values with the key "interpolation" that contain a string
    to the corresponding interpolation.

    Args:
        cfg (dict): The config to parse.

    Returns:
        dict: The parsed config.
    """

    transform_cfg = cfg.get("transforms", {})
    interpolation_keys = ["interpolation", "img_interpolation", "mask_interpolation"]
    for split_type_key, split_transform_list in transform_cfg.items():
        if isinstance(split_transform_list, list):
            for i, transform_dict in enumerate(split_transform_list):
                if isinstance(transform_dict, dict):
                    for transform_key, transform_args in transform_dict.items():
                        if isinstance(transform_args, dict):
                            for interpolation_key in interpolation_keys:
                                if interpolation_key in transform_args:
                                    interpolation_value = transform_args[interpolation_key].lower()
                                    if interpolation_value not in INTERPOLATION_MAPPINGS:
                                        raise ValueError(f'Interpolation {interpolation_value} is not supported')
                                    cfg["transforms"][split_type_key][i][transform_key][interpolation_key] \
                                        = INTERPOLATION_MAPPINGS[interpolation_value]
    return cfg


@beartype
def parse_metrics(cfg: dict) -> dict:
    """
    Parses metrics in the config from strings to the EvalMetric enum. Also parses the metric averaging method
    from a string to the MetricsMethod enum.

    Args:
        cfg (dict): The config dictionary.

    Returns:
        dict: The parsed config dictionary.
    """

    if "metrics" not in cfg or not isinstance(cfg["metrics"], dict):
        return cfg
    for key, value in cfg["metrics"].items():
        if key == "tracked":
            for i, metric in enumerate(value):
                if isinstance(metric, str):
                    cfg["metrics"][key][i] = EvalMetric(metric)
        elif key == "averaging_method":
            cfg["metrics"][key] = MetricsMethod(value)
    return cfg


@beartype
def parse_hyperopt_cfg(cfg: dict) -> dict:
    """
    Parses the hyperopt config in the config from strings to the corresponding hyperopt parameter space.

    Args:
    cfg (dict): The config dictionary.

    Returns:
        dict: The parsed config dictionary.
    """
    if "hyperopt" in cfg and "param_space" in cfg["hyperopt"]:
        if cfg["hyperopt"]["type"]=="grid_search":
            # Do not parse the dict with hyperopt package; leave as it is
            pass
        else:
            cfg["hyperopt"]["param_space"] = parse_hyperopt_param_space(cfg["hyperopt"]["param_space"])
    return cfg


@beartype
def parse_hyperopt_param_space(param_dict: dict):
    """
    Parses the hyperopt parameter space in the config from strings to the corresponding hyperopt parameter space.

    Args:
        param_dict (dict): The parameter space dictionary.

    Returns:
        dict: The parsed parameter space dictionary.
    """

    for key, value in param_dict.items():
        if isinstance(value, dict):
            param_dict[key] = parse_hyperopt_param_space(value)
        elif isinstance(value, list):
            param_dict[key] = hp.choice(key, value)
    return param_dict


@beartype
def parse_checkpoint_cfg(cfg: dict):
    """
    Parses the checkpoint settings in the config from strings to the corresponding boolean values.

    Args:
        cfg (dict): The config dictionary.

    Returns:
        dict: The parsed config dictionary.
    """
    if 'settings' not in cfg.keys():
        return cfg
    checkpoints_cfg = cfg['settings'].get('checkpoints', None)
    if checkpoints_cfg is None:
        # maintain backwards compatibility to old configs
        # TODO: can be removed if old configs and checkpoints are not used anymore
        save_checkpoints = cfg['settings'].pop('save_checkpoints', False)
        save_best_checkpoint_only = cfg['settings'].pop('save_best_checkpoint_only', False)
        save_best_checkpoint_metric = EvalMetric(
            cfg['settings'].pop('save_best_checkpoint_metric', EvalMetric.DICE.value))
        save_checkpoints_above_epoch = cfg['settings'].pop('save_checkpoints_above_epoch', None)

        checkpoints_cfg = dict()
        checkpoints_cfg['save_mode'] = 'none'
        if save_checkpoints:
            checkpoints_cfg['save_mode'] = 'last'
            if save_best_checkpoint_only:
                checkpoints_cfg['save_mode'] = 'best'
                checkpoints_cfg['metric'] = save_best_checkpoint_metric.value
        if save_checkpoints_above_epoch is not None:
            # in old configs, save_checkpoints_above_epoch superseded save_checkpoints setting
            checkpoints_cfg['save_mode'] = 'last' if checkpoints_cfg['save_mode'] == 'none' else checkpoints_cfg[
                'save_mode']
            checkpoints_cfg['metric'] = save_best_checkpoint_metric
            checkpoints_cfg['min_epoch'] = save_checkpoints_above_epoch

        cfg['settings']['checkpoints'] = checkpoints_cfg

    assert checkpoints_cfg['save_mode'] in ['best', 'last', 'best_and_last', 'none']
    if checkpoints_cfg['save_mode'] == 'best':
        assert 'metric' in checkpoints_cfg
    if checkpoints_cfg['save_mode'] == 'best_and_last':
        assert 'metric' in checkpoints_cfg
    if 'min_epoch' not in checkpoints_cfg:
        checkpoints_cfg['min_epoch'] = 0
    assert isinstance(checkpoints_cfg['min_epoch'], int)
    if 'metric' in checkpoints_cfg and isinstance(checkpoints_cfg['metric'], str):
        checkpoints_cfg['metric'] = EvalMetric(checkpoints_cfg['metric'])

    cfg['settings']['checkpoints'] = checkpoints_cfg
    return cfg
