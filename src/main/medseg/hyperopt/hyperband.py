import math
import os
import pickle
import shutil
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from hyperopt.pyll.stochastic import sample as hyperopt_sample
from numpy.random import Generator, PCG64

from medseg.config.config import merge_configs
from medseg.data.split_type import SplitType
from medseg.evaluation.metrics import EvalMetric
from medseg.training.trainer import Trainer
from medseg.util.date_time import get_current_date_time_str
from medseg.util.files import save_text_to_file
from medseg.util.logger import flatten_dict, create_custom_logger
from medseg.util.path_builder import PathBuilder

DEFAULT_R = 81
DEFAULT_ETA = 3


class HyperbandOptimizer:
    """
    Hyperband optimizer (in conjunction with random search) for hyperparameter optimization. For more information,
    see the original publication https://arxiv.org/abs/1603.06560. The variable names used here are the same as
    described in the publication.
    """

    def __init__(self, cfg: dict, cfg_path: str = None, simulate: bool = False):
        self.cfg = cfg
        model_name = self.cfg['architecture'].get('model_name', '')
        self.hyperopt_name = cfg.get('hyperopt_name', f"{model_name}_hyperband_{get_current_date_time_str()}")
        self.cfg['hyperopt_name'] = self.hyperopt_name
        self.save_path_builder = PathBuilder(self.cfg).root().out().hyperopt_runs().hyperopt_name()
        if cfg_path is not None:
            shutil.copy(cfg_path, self.save_path_builder.clone().add(os.path.basename(cfg_path)).build())
        self.param_space = cfg['hyperopt']['param_space']
        self.R = cfg['hyperopt'].get('R', DEFAULT_R)
        self.eta = cfg['hyperopt'].get('eta', DEFAULT_ETA)
        self.maximize = cfg['hyperopt']['maximize']
        self.metric = cfg['hyperopt']['metric'].value if isinstance(cfg['hyperopt']['metric'], EvalMetric) \
            else cfg['hyperopt']['metric'].lower()
        assert self.metric == 'loss' or self.metric in [m.value for m in EvalMetric]
        self.s_max = math.floor(math.log(self.R, self.eta))  # maximum number of iterations per bracket
        self.B = (self.s_max + 1) * self.R  # total resources allocated
        self.T = list()  # list of param spaces that are currently tried
        self.is_resumed_from_state = False
        self.current_iter = {'s': -1, 'i': -1, 'c': -1}
        self.results = []
        bit_generator = PCG64(self.cfg['hyperopt'].get('random_seed', 42))
        self.rng = Generator(bit_generator)
        self.last_state = self.rng.bit_generator.state
        self.simulate = simulate
        log_path = self.save_path_builder.clone().add('hyperopt_log.txt').build()
        self.logger = create_custom_logger(self.hyperopt_name, log_path)
        self.trial_param_space_dict = OrderedDict()
        self.use_previous_checkpoints = cfg['hyperopt'].get('use_previous_checkpoints', False)
        if self.use_previous_checkpoints:
            self.logger.warning("The use_previous_checkpoints flag was set to True. This does not restore any random "
                                "states from the loaded checkpoints, particularly not from dataloaders, causing the "
                                "training to be"
                                "non-deterministic independently from the full determinism setting. It might also "
                                "degrade neural net performance as randomly sampled parameters in augmentations will "
                                "be identical to the ones used in the respective restored checkpoint.")

    @classmethod
    def from_state_dict(cls, state_dict: dict, simulate: bool = False) -> 'HyperbandOptimizer':
        hb_optimizer = cls(state_dict['cfg'], simulate=simulate)
        hb_optimizer.is_resumed_from_state = True
        if 's_max' in state_dict:
            hb_optimizer.s_max = state_dict['s_max']
        if 'B' in state_dict:
            hb_optimizer.B = state_dict['B']
        if 'results' in state_dict:
            hb_optimizer.results = state_dict['results']
        if 'R' in state_dict:
            hb_optimizer.R = state_dict['R']
        if 'T' in state_dict:
            hb_optimizer.T = state_dict['T']
        if 'current_iter' in state_dict:
            hb_optimizer.current_iter = state_dict['current_iter']
        if 'rng_state' in state_dict:
            hb_optimizer.rng.bit_generator.state = state_dict['rng_state']
            hb_optimizer.last_state = state_dict['rng_state']
        if 'trial_param_space_dict' in state_dict:
            hb_optimizer.trial_param_space_dict = state_dict['trial_param_space_dict']
        return hb_optimizer

    def save_state(self) -> None:
        state = {
            'cfg': self.cfg,
            's_max': self.s_max,
            'B': self.B,
            'R': self.R,
            'T': self.T,
            'results': self.results,
            'rng_state': self.last_state,
            'current_iter': self.current_iter,
            'trial_param_space_dict': self.trial_param_space_dict
        }
        file_path = self.save_path_builder.clone().add('hyperopt_state.pkl').build()
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    def try_to_load_checkpoint(self, param_space: dict, trial_cfg: dict) -> Tuple[Optional[Dict], Optional[str]]:
        previous_trial_name = self.find_latest_trial_with_param_space(param_space)
        if previous_trial_name is not None:
            path = self.try_to_find_checkpoint(previous_trial_name, trial_cfg['trial_bracket'])
            checkpoint = torch.load(path)
            epochs_to_train = trial_cfg['settings']['max_epochs']
            if checkpoint['cfg']['trial_param_space'] != param_space:
                print(f"WARNING: param space of checkpoint {path} seems to not match the current param space.")
            if checkpoint['cfg']['settings']['max_epochs'] > epochs_to_train:
                # if the checkpoint has been trained on more epochs it can't be used for resumption. In theory this
                # should never happen, but just to be sure we check it here.
                return None, None
            else:
                # update cfg with trial name, etc... and return checkpoint
                checkpoint['cfg'] = merge_configs(trial_cfg, checkpoint['cfg'])
                self.logger.info(f"Resuming from previous checkpoint under {path}")
                return checkpoint, path
        return None, None

    def try_to_find_checkpoint(self, trial_name: str, trial_bracket: str) -> Optional[str]:
        cfg_for_pathbuilder = deepcopy(self.cfg)
        cfg_for_pathbuilder['trial_name'] = trial_name
        cfg_for_pathbuilder['trial_bracket'] = trial_bracket
        previous_trial_path_builder = PathBuilder.trial_out_builder(cfg_for_pathbuilder)
        if os.path.exists(previous_trial_path_builder.clone().build()):
            filenames = ['final_checkpoint.pt', 'latest_checkpoint.pt']
            for filename in filenames:
                checkpoint_path = previous_trial_path_builder.clone().add(filename).build()
                if os.path.exists(checkpoint_path):
                    return checkpoint_path
        return None

    def find_latest_trial_with_param_space(self, param_space: dict) -> Optional[str]:
        for trial_name, trial_param_space in reversed(self.trial_param_space_dict.items()):
            if trial_param_space == param_space:
                return trial_name
        return None

    def run(self) -> Tuple[Dict[str, Any], float, str]:
        summary = self.get_summary()
        print(summary)
        self.save_to_txt(summary, 'hyperopt_summary.txt')

        start_s = self.s_max
        for s in reversed(range(start_s + 1)):
            if self.is_resumed_from_state and self.current_iter['s'] is not None and s > self.current_iter['s']:
                continue  # Restore iteration if necessary
            n, r, T = self._initialize_bracket(s)
            if not self.is_resumed_from_state or len(self.T) == 0:
                self.T = T
            self.current_iter['s'] = s
            self.log_bracket_info(s, n, r)
            self._iterate_bracket(s, n, r)
            self.current_iter['i'] = -1  # reset i after bracket is finished

        best_trial_name, best_metric, best_config = sorted(self.results, key=lambda x: x[1], reverse=self.maximize)[0]
        return best_trial_name, best_metric, best_config

    def _initialize_bracket(self, s: int) -> Tuple[int, int, List[Dict[str, Any]]]:
        """Initializes a bracket for the Hyperband algorithm."""
        n = self._calculate_n(s)
        r = self._calculate_r(s)
        self.last_state = self.rng.bit_generator.state
        T = [hyperopt_sample(self.param_space, rng=self.rng) for _ in range(n)]
        return n, r, T

    def _calculate_n(self, s: int) -> int:
        return int(np.ceil((self.B / self.R) * (self.eta ** s) / (s + 1)))

    def _calculate_r(self, s: int) -> int:
        return self.R * self.eta ** (-s)

    def _iterate_bracket(self, s: int, n: int, r: int) -> None:
        """
        Iterate through the bracket. This is the inner loop of Hyperband, which is equivalent to
        successive halving.
        """
        for i in range(s + 1):
            if self.is_resumed_from_state and self.current_iter['i'] is not None and i < self.current_iter['i']:
                continue  # Restore iteration if necessary
            self.current_iter['i'] = i

            n_i = int(n * self.eta ** (-i))  # Determine number of configs evaluated in this iteration
            r_i = int(r * self.eta ** i)  # Determine resource allocation for each config
            self.log_bracket_iteration_info(s, i, n_i, r_i)
            L = self._iterate_configs(r_i)
            self.current_iter['c'] = -1  # Reset k iteration after finishing the bracket
            indices = np.argsort(L)
            if self.maximize:
                indices = indices[::-1]  # reverse
            indices = indices[:int(n_i / self.eta)]  # keep only the n_i/eta best configurations
            self.T = [self.T[j] for j in indices]
            self.save_state()

    def _iterate_configs(self, r_i: int):
        """Iterate through the configurations in the bracket."""

        total_t = len(self.T)
        metric_values = []
        for c, t in enumerate(self.T):
            # Restore iteration if necessary
            if self.is_resumed_from_state and self.current_iter['c'] is not None and c < self.current_iter['c']:
                continue
            self.current_iter['c'] = c
            self.save_state()
            flattened_cfg = flatten_dict(t, sep=' -> ')
            self.log_config_info(c, r_i, total_t, flattened_cfg)

            trial_cfg = deepcopy(self.cfg)
            trial_cfg['trial_param_space'] = t
            trial_cfg['trial_bracket'] = self.current_iter['s']
            trial_cfg['trial_bracket_iteration'] = self.current_iter['i']
            trial_cfg['trial_bracket_config'] = c
            trial_cfg['settings']['max_epochs'] = r_i
            trial_id = f"r{r_i:04d}_s{self.current_iter['s']:02d}_i{self.current_iter['i']:02d}_c{c:04d}"
            trial_cfg['trial_name'] = f"{trial_id}_{get_current_date_time_str()}"

            if not self.simulate:
                trainer = None
                previous_checkpoint_path = None
                if self.use_previous_checkpoints:
                    # Check if trial with same param space already exists and use its checkpoint for the new one
                    checkpoint_to_resume, previous_checkpoint_path = self.try_to_load_checkpoint(t, trial_cfg)
                    if checkpoint_to_resume is not None:
                        trainer = Trainer.from_state_dict(checkpoint_to_resume)

                if trainer is None: trainer = Trainer(trial_cfg)
                trainer.train()

                if previous_checkpoint_path is not None: os.remove(previous_checkpoint_path)

                if self.metric == 'loss':
                    metric_value = trainer.state.metrics_manager.get_last_mean_loss(SplitType.VAL)
                else:
                    metric_value = trainer.state.metrics_manager.get_last_metric(SplitType.VAL, EvalMetric(self.metric))
                trainer.free_memory()
                del trainer

            else:
                # Simulation mode, so we generate a random metric value
                metric_value = np.random.uniform(0, 1)

            self.results.append((trial_cfg['trial_name'], metric_value, deepcopy(t)))
            self.trial_param_space_dict[trial_cfg['trial_name']] = deepcopy(t)
            metric_values.append(metric_value)
            self.save_state()
            self.save_to_txt(self.get_summary(), 'hyperopt_summary.txt')
        return metric_values

    def save_to_txt(self, content: str, file_name: str) -> None:
        save_text_to_file(content, self.save_path_builder.clone().add(file_name).build())

    def get_summary(self) -> str:
        summary = "Hyperband Run Summary\n"
        summary += "============================================\n\n"
        summary += f"Maximum resources allocated per trial (R): {self.R}\n"
        summary += f"Reduction factor (eta): {self.eta}\n"
        summary += f"Brackets (from s_max to 0): {','.join(map(str, reversed(range(self.s_max + 1))))}\n"
        summary += f"Budget (B) - approx. resources (epochs) allocated per bracket: {self.B}\n"
        summary += f"Optimization metric: validation {self.metric}\n"
        summary += f"Optimization direction: {'maximize' if self.maximize else 'minimize'}\n"
        summary += f"Total number of trials: {len(self.results)}\n\n"
        summary += "Brackets overview\n"
        summary += "---------------\n\n"
        summary += "  s = Bracket number (starts with highest)\n"
        summary += "  i = Iteration within bracket\n"
        summary += "  n_i = Number of configs in iteration\n"
        summary += "  r_i = Amount of resources per config (epochs)\n\n"
        summary += "     |"
        for s in reversed(range(self.s_max + 1)):
            summary += f"    s={s}    |"
        summary += "\n     |"
        for _ in reversed(range(self.s_max + 1)):
            summary += "           |"
        summary += "\n   i |"
        for _ in reversed(range(self.s_max + 1)):
            summary += " n_i | r_i |"
        summary += "\n   - |"
        for _ in reversed(range(self.s_max + 1)):
            summary += " --- | --- |"
        summary += "\n"

        total_r = 0
        for i in range(self.s_max + 1):
            summary += f"   {i} |"
            for s in reversed(range(self.s_max + 1)):
                if i <= s:
                    n = self._calculate_n(s)
                    r = self._calculate_r(s)
                    n_i = int(n * self.eta ** (-i))
                    r_i = int(r * self.eta ** i)
                    total_r += n_i * r_i
                    summary += f" {n_i:<3} | {r_i:<3} |"
                else:
                    summary += "     |     |"
            summary += "\n"
        summary += f"Total resources (epochs) allocated: {total_r}\n\n"

        if len(self.results) > 0:
            results_sorted = sorted(self.results, key=lambda x: x[1], reverse=self.maximize)
            summary += "\nConfigurations (sorted by best metric descending):\n"
            summary += "---------------\n"
            for idx, (trial_name, metric, cfg) in enumerate(results_sorted):
                summary += f"   Index: {idx},  Trial name: {trial_name}\n"
                summary += f"    Metric: {metric}\n"
                summary += "    Configuration:\n"
                for k, v in flatten_dict(cfg, sep=' -> ').items():
                    summary += f"      {k}: {v}\n"
                summary += "\n"
        summary += "============================================\n"
        return summary

    def log_bracket_info(self, s: int, n: int, r: int) -> None:
        self.logger.info("============================================")
        self.logger.info(f"Bracket {s} (counting down until bracket 0):")
        self.logger.info(f"  Number of initial random configurations (n): {n}")
        self.logger.info(f"  Initial resource allocation per configuration (r): {r}")
        self.logger.info("============================================")

    def log_bracket_iteration_info(self, s: int, i: int, n_i: int, r_i: int) -> None:
        self.logger.info("============================================")
        self.logger.info(f"  Starting bracket {s} iteration {i}:")
        self.logger.info(f"    Number of configurations (n_i): {n_i}")
        self.logger.info(f"    Resource allocation per configuration (r_i): {r_i}")
        self.logger.info("============================================")

    def log_config_info(self, c: int, r_i: int, total_t: int, flattened_cfg: Dict[str, Any]) -> None:
        self.logger.info("============================================")
        self.logger.info(
            f"Starting training for config number (c) {c + 1}/{total_t} for {r_i} epochs with the following params:")
        for k, v in flattened_cfg.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("============================================")
