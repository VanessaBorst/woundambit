import copy

import oyaml

from medseg.config.parse import parse_interpolation, parse_all_tuples, parse_metrics, \
    parse_hyperopt_cfg, parse_checkpoint_cfg
from medseg.config.parse_transforms import parse_transforms


# TODO define static dict with default config values, use method to fill in missing values if config loaded from
#  file, print info that default values are used
def merge_dicts(priority_dict, other_dict):
    for key, value in priority_dict.items():
        if key in other_dict and isinstance(value, dict) and isinstance(other_dict[key], dict):
            merge_dicts(value, other_dict[key])
        elif key in other_dict and isinstance(value, list) and isinstance(other_dict[key], list):
            merge_lists(value, other_dict[key])
        else:
            other_dict[key] = value


def merge_lists(priority_list, other_list):
    for index, value in enumerate(priority_list):
        if index < len(other_list):
            if isinstance(value, dict) and isinstance(other_list[index], dict):
                merge_dicts(value, other_list[index])
            elif isinstance(value, list) and isinstance(other_list[index], list):
                merge_lists(value, other_list[index])
            else:
                other_list[index] = value
        else:
            other_list.append(value)


def merge_configs(priority_cfg: dict, other_cfg: dict) -> dict:
    """
    Merge two config dicts. Values from priority_cfg overwrite values from other_cfg.
    To merge, loops through all values in tune_cfg, including the ones in nested dicts and lists at any level.
    Every time an "end node" is encountered (i.e. a value that is not a dict or list), it is added to the config
    """
    result_cfg = copy.deepcopy(other_cfg)
    merge_dicts(priority_cfg, result_cfg)
    return result_cfg


def load_and_parse_config(file_path: str) -> dict:
    with open(file_path) as f:
        cfg = oyaml.safe_load(f)
        cfg = parse_metrics(cfg)
        cfg = parse_all_tuples(cfg)
        cfg = parse_interpolation(cfg)
        cfg = parse_transforms(cfg)
        cfg = parse_hyperopt_cfg(cfg)
        cfg = parse_checkpoint_cfg(cfg)
        return cfg


def is_hyperopt_run(cfg: dict) -> bool:
    return 'hyperopt' in cfg \
           and cfg['hyperopt'] is not False \
           and 'type' in cfg['hyperopt'] \
           and 'param_space' in cfg['hyperopt'] \
           and len(cfg['hyperopt']['param_space']) > 0


def is_hyperband_run(cfg: dict) -> bool:
    return is_hyperopt_run(cfg) and cfg['hyperopt']['type'].lower() == 'hyperband'


# def is_grid_search_run(cfg: dict) -> bool:
#     return is_hyperopt_run(cfg) and cfg['hyperopt']['type'].lower() == 'grid_search'


def is_k_fold_run(cfg: dict) -> bool:
    return 'k_fold' in cfg \
           and cfg['k_fold'] is not False \
           and 'k' in cfg['k_fold'] \
           and cfg['k_fold']['k'] > 0
