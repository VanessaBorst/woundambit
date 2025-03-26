from itertools import product
import os
import pickle
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from medseg.config.config import load_and_parse_config
from medseg.data.split_type import SplitType
from medseg.evaluation.metrics import EvalMetric
from medseg.training.trainer import Trainer
from medseg.util.date_time import get_current_date_time_str
from medseg.util.files import save_text_to_file
from medseg.util.helper_functions import nested_get
from medseg.util.logger import flatten_dict, create_custom_logger
from medseg.util.param_study_ops import encode_trial_name_short
from medseg.util.path_builder import PathBuilder


def generate_combinations(param_space):
    """
    Generate all combinations of parameters while maintaining nested structure.

    Args:
        param_space (dict): Nested parameter space.

    Returns:
        list: A list of all possible configurations.
    """

    def recursive_combinations(space):
        if isinstance(space, dict):
            # Recursively compute combinations for nested dictionaries
            keys = space.keys()
            combinations = product(*[recursive_combinations(space[k]) for k in keys])
            return [dict(zip(keys, comb)) for comb in combinations]
        elif isinstance(space, list):
            # Return list of values directly for leaf nodes
            return space
        else:
            raise ValueError("Unsupported type in parameter space. Only dict and list are allowed.")

    return recursive_combinations(param_space)


def update_config(config, trial_params):
    """
    Recursively updates the config dictionary with trial-specific parameters.

    Args:
        config (dict): Original configuration dictionary.
        trial_params (dict): Trial-specific parameters to update.

    Returns:
        dict: Updated configuration dictionary.
    """
    for key, value in trial_params.items():
        if isinstance(value, dict) and key in config:
            # Recursively update nested dictionaries
            config[key] = update_config(config[key], value)
        else:
            # Update value directly
            config[key] = value
    return config

class GridSearchOptimizer:
    """
    GridSearch optimizer for hyperparameter optimization.
    It will evaluate all combinations of parameters in a grid search.
    """

    def __init__(self, cfg: dict, cfg_path: str = None, simulate: bool = False, is_resumed_from_state: bool = False):
        self.cfg = cfg
        model_name = self.cfg['architecture'].get('model_name', '')

        if not is_resumed_from_state:
            self.hyperopt_name = cfg.get('hyperopt_name', f"{model_name}_gridsearch_{get_current_date_time_str()}")
        else:
            self.hyperopt_name = nested_get(cfg, ['hyperopt', 'resume_folder'],
                                            default=f"{model_name}_gridsearch_{get_current_date_time_str()}")
            print(f"Resumed from {self.hyperopt_name}")
        self.cfg['hyperopt_name'] = self.hyperopt_name
        self.save_path_builder = PathBuilder(self.cfg).root().out().hyperopt_runs().hyperopt_name()
        print(f"Saving results to {self.save_path_builder.build()}")
        if cfg_path is not None:
            shutil.copy(cfg_path, self.save_path_builder.clone().add(os.path.basename(cfg_path)).build())

        self.param_space = cfg['hyperopt']['param_space']
        self.maximize = cfg['hyperopt']['maximize']
        self.metric = cfg['hyperopt']['metric'].value if isinstance(cfg['hyperopt']['metric'], EvalMetric) \
            else cfg['hyperopt']['metric'].lower()
        assert self.metric == 'loss' or self.metric in [m.value for m in EvalMetric]


        self.results = []
        self.simulate = simulate
        log_path = self.save_path_builder.clone().add('hyperopt_log.txt').build()
        self.logger = create_custom_logger(self.hyperopt_name, log_path)


    @classmethod
    def resume_from_path(cls, path: str, simulate: bool = False) -> 'GridSearchOptimizer':
        cfg = load_and_parse_config(path)
        cfg['hyperopt']['resume_folder'] = str(Path(path).parent.name)
        gs_optimizer = cls(cfg=cfg, simulate=simulate, is_resumed_from_state=True)
        gs_optimizer.logger.info(
            f"Resuming grid search from {path}. This will skip any previously completed trials and resume from the "
            f"latest checkpoint of the last running trial before the interruption."
        )

        return gs_optimizer

    def save_state(self) -> None:
        state = {
            'cfg': self.cfg,
            'results': self.results
        }
        file_path = self.save_path_builder.clone().add('hyperopt_gridsearch_state.pkl').build()
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    def run(self) -> Tuple[Dict[str, Any], float, str]:
        """
        Run the GridSearch by evaluating all possible combinations of hyperparameters,
        with support for resuming interrupted runs and individual trial checkpoint recovery.
        """

        # Attempt to load saved state
        file_path = self.save_path_builder.clone().add('hyperopt_gridsearch_state.pkl').build()
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                saved_state = pickle.load(f)
                self.cfg = saved_state['cfg']
                self.results = saved_state['results']
                self.logger.info("Loaded saved state for resuming grid search.")
        else:
            self.logger.info(f"No saved state found at {file_path}. Starting fresh grid search.")

        summary = self.get_summary()
        print(summary)
        self.save_to_txt(summary, 'hyperopt_summary.txt')

        # Generate all combinations of hyperparameters
        combinations = generate_combinations(self.param_space)
        total_combinations = len(combinations)
        self.logger.info(f"Total combinations: {total_combinations}")

        # Skip already completed combinations
        completed_combinations = {tuple(flatten_dict(cfg).items()) for _, _, cfg in self.results}
        self.logger.info(f"Skipping {len(completed_combinations)} already completed trials.")

        best_trial_name = None
        best_metric = None
        best_config = None


        for param_comb in combinations:

            # Skip if this combination has already been evaluated
            flat_comb = tuple(flatten_dict(param_comb).items())
            if flat_comb in completed_combinations:
                self.logger.info(f"Skipping already completed combination: {param_comb}")
                continue

            self.logger.info(f"Evaluating configuration: {param_comb}")

            # Prepare the trial configuration
            trial_cfg = deepcopy(self.cfg)
            trial_cfg['trial_param_space'] = param_comb
            trial_cfg = update_config(trial_cfg, param_comb)
            trial_cfg['settings']['max_epochs'] = trial_cfg['settings'].get('max_epochs', 100)
            trial_name = encode_trial_name_short(param_comb)
            trial_cfg['trial_name'] = trial_name

            # Set checkpoint path for resuming trials
            trial_checkpoint_path = PathBuilder.trial_out_builder(trial_cfg).add('latest_checkpoint.pt').build()

            if not self.simulate:

                # Check if a checkpoint exists for this trial
                if os.path.exists(trial_checkpoint_path):
                    self.logger.info(f"Resuming trial {trial_name} from checkpoint {trial_checkpoint_path}.")
                    checkpoint = torch.load(trial_checkpoint_path)
                    trainer = Trainer.from_state_dict(checkpoint)
                    # Manually trigger the scheduler update if the scheduler is not None (for learning rate adaptation)
                    if trainer.state.scheduler is not None:
                        trainer.update_scheduler()
                else:
                    trainer = Trainer(trial_cfg)

                trainer.train()

                if self.metric == 'loss':
                    metric_value = trainer.state.metrics_manager.get_last_mean_loss(SplitType.VAL)
                else:
                    metric_value = trainer.state.metrics_manager.get_last_metric(SplitType.VAL, EvalMetric(self.metric))
                trainer.free_memory()
                del trainer
            else:
                # Simulation mode, so we generate a random metric value
                metric_value = np.random.uniform(0, 1)

            self.results.append((trial_cfg['trial_name'], metric_value, deepcopy(param_comb)))
            self.save_state()
            self.save_to_txt(self.get_summary(), 'hyperopt_summary.txt')

            # Track the best result
            if best_metric is None or (self.maximize and metric_value > best_metric) or (not self.maximize and metric_value < best_metric):
                best_trial_name = trial_cfg['trial_name']
                best_metric = metric_value
                best_config = param_comb

        return best_trial_name, best_metric, best_config

    def save_to_txt(self, content: str, file_name: str) -> None:
        save_text_to_file(content, self.save_path_builder.clone().add(file_name).build())

    def get_summary(self) -> str:
        summary = "GridSearch Run Summary\n"
        summary += "============================================\n\n"
        summary += f"Optimization metric: validation {self.metric}\n"
        summary += f"Optimization direction: {'maximize' if self.maximize else 'minimize'}\n"
        summary += f"Total number of trials: {len(self.results)}\n\n"
        summary += "Configurations evaluated:\n"
        summary += "---------------\n"
        results_sorted = sorted(self.results, key=lambda x: x[1], reverse=self.maximize)
        for idx, (trial_name, metric, cfg) in enumerate(results_sorted):
            summary += f"   Index: {idx},  Trial name: {trial_name}\n"
            summary += f"    Metric: {metric}\n"
            summary += "    Configuration:\n"
            for k, v in flatten_dict(cfg, sep=' -> ').items():
                summary += f"      {k}: {v}\n"
            summary += "\n"
        summary += "============================================\n"
        return summary

