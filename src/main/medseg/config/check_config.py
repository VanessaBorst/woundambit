from typing import List, Tuple, Optional

REQUIRED_ARCHITECTURE_KEYS = [
    "arch_type",
]

REQUIRED_DATASET_KEYS = [
    "type",
]

REQUIRED_SETTINGS_KEYS = [
    "max_epochs",
    "batch_size",
    "random_seed",
    "num_workers",
    "mixed_precision",
    "torch_compile",
]

REQUIRED_OPTIMIZER_KEYS = [
    "type",
    "lr"
]

REQUIRED_SCHEDULER_KEYS = [
    "type",
]

REQUIRED_TRANSFORMS_KEYS = [

]

REQUIRED_HYPEROPT_KEYS = [
    "metric",
    "mode",
    "param_space"
]

EITHER_OR_KEYS_SETTINGS = [
    ("max_epochs", "max_iterations")
]

UNIQUE_TRANSFORMS = [
    'normalize_zscore'
]

CAN_BE_FALSE = [
    "scheduler",
]

def check_config(cfg: dict):
    """
    Check if the loaded config is valid.
    """
    # TODO: check metrics cfg
    # TODO: check early stop cfg
    # TODO: check if in_size and pad_and_resize are compatible
    # TODO: check if Normalize comes before NormalizeZScore
    # TODO: check if NormalizeZScore is twice
    # TODO: check unique transforms
    # TODO: warning if there is no to_tensor transform
    # TODO: check transforms against transform backend
    # TODO: warning if several of the same datasets are used or add support for this (dataset.get_name() is not unique
    #  in that case, which potentially causes problems at different locations in the code)
    main_cfg = cfg
    arch_cfg = cfg["architecture"]
    settings_cfg = cfg["settings"]
    dataset_cfg = cfg["dataset"]
    optimizer_cfg = cfg["optimizer"]
    scheduler_cfg = cfg["scheduler"]
    transforms_cfg = cfg["transforms"]

    check_if_dict_and_nonempty(main_cfg, "Main")
    check_subconfig(arch_cfg, "Architecture", REQUIRED_ARCHITECTURE_KEYS)
    check_subconfig(settings_cfg, "Settings", REQUIRED_SETTINGS_KEYS, EITHER_OR_KEYS_SETTINGS)
    check_subconfig(dataset_cfg, "Dataset", REQUIRED_DATASET_KEYS)
    check_subconfig(optimizer_cfg, "Optimizer", REQUIRED_OPTIMIZER_KEYS)
    check_subconfig(scheduler_cfg, "Scheduler", REQUIRED_SCHEDULER_KEYS)
    check_subconfig(transforms_cfg, "Transforms", REQUIRED_TRANSFORMS_KEYS)
    check_transforms(cfg)
    if "tune" in cfg:
        tune_cfg = cfg["tune"]
        check_subconfig(tune_cfg, "Tune", REQUIRED_HYPEROPT_KEYS)


def check_subconfig(cfg: dict, name: str, required_keys: List[str],
                    either_or_keys: Optional[List[Tuple[str, str]]] = None):
    """
    Check if the loaded dataset config is valid.
    """
    if isinstance(cfg, bool) and cfg is False and name.lower() in CAN_BE_FALSE:
        return
    check_if_dict_and_nonempty(cfg, name)
    check_required_keys(cfg, name, required_keys)
    if either_or_keys is not None:
        check_either_or_keys(cfg, name, either_or_keys)


def check_either_or_keys(cfg: dict, name: str, either_or_keys: List[Tuple[str, str]]):
    """Check if either one of the keys in keys1 or keys2 is present in the config."""
    for key1, key2 in either_or_keys:
        if key1 not in cfg and key2 not in cfg:
            raise ValueError(f"Either {key1} or {key2} must be present in {name}-config.")
        elif key1 in cfg and key2 in cfg:
            raise ValueError(f"Both {key1} and {key2} are present in {name}-config. Only one of them is allowed.")


def check_required_keys(cfg: dict, name: str, required_keys: List[str]):
    """Check if all required keys are present in the config."""
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"Required key {key} not found in {name}-config.")


def check_if_dict_and_nonempty(cfg: dict, name: str):
    """Check if the loaded config is a dictionary."""
    if cfg is None or not isinstance(cfg, dict):
        raise TypeError(f"{name}-config must be a dict, but got {type(cfg)}")
    elif len(cfg) == 0:
        raise ValueError(f"{name}-config must not be empty")


def check_transforms(cfg: dict):
    """ Checks if the transforms config is valid. In particular, checks if indents were done wrong.
     This can lead to the transforms being in the same dictionary as the arguments.
     """

    transform_cfg = cfg["transforms"]
    for split_type_key, split_transform_list in transform_cfg.items():
        if isinstance(split_transform_list, list):
            for i, transform_dict in enumerate(split_transform_list):
                if isinstance(transform_dict, dict):
                    if len(transform_dict) != 1:
                        raise ValueError(f"Transform {i + 1} in {split_type_key} has an invalid format, likely due to "
                                         f"missing indentation of transform arguments. Expected a single key-value "
                                         f"pair, but got {transform_dict}")
