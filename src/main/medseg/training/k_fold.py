import glob
import os
import pickle
import shutil
from copy import deepcopy
from typing import Optional, List, Tuple

import torch
from beartype import beartype

from medseg.data.dataset_manager import DatasetManager
from medseg.data.split_type import SplitType
from medseg.evaluation.ensemble_evaluator import EnsembleEvaluator
from medseg.training.trainer import Trainer
from medseg.util.date_time import get_current_date_time_str
from medseg.util.files import find_file, find_first_file_with_extension
from medseg.util.logger import create_custom_logger
from medseg.util.path_builder import PathBuilder
from medseg.util.random import get_folds


class KFoldCrossValidator:
    @beartype
    def __init__(self, cfg: dict, cfg_or_dir_path: Optional[str] = None, is_resumed_from_state: bool = False):

        assert "k_fold" in cfg, "k_fold config not found"

        self.cfg = cfg

        if cfg_or_dir_path is not None:
            if cfg_or_dir_path.endswith(".pkl"):
                cfg_or_dir_path = os.path.dirname(cfg_or_dir_path)
            if not cfg_or_dir_path.endswith(".yaml"):
                cfg_or_dir_path = find_first_file_with_extension(cfg_or_dir_path, ".yaml")
        self.cfg_path = cfg_or_dir_path
        self.save_pb = None
        self.k_fold_name = cfg["k_fold"].get('k_fold_name', None)
        if self.cfg_path is not None:
            if is_resumed_from_state:
                dir_path = os.path.dirname(self.cfg_path)
                self.save_pb = PathBuilder.from_path(dir_path, deepcopy(self.cfg))
                self.k_fold_name = os.path.basename(dir_path)

        if self.k_fold_name is None:
            self.k_fold_name = self._get_k_fold_folder_name()
        self.cfg["k_fold"]["k_fold_name"] = self.k_fold_name
        if self.save_pb is None:
            self.save_pb = PathBuilder(self.cfg).root().out().k_fold().add(self.k_fold_name)
        if self.cfg_path is not None and self.cfg_path.endswith(".yaml"):
            target_path = self.save_pb.clone().add(os.path.basename(self.cfg_path)).build()
            if self.cfg_path != target_path:
                shutil.copy(self.cfg_path, target_path)

        self.k = cfg["k_fold"]["k"]
        self.random_seed = cfg["k_fold"].get('random_seed', None)
        self.include_test_split = cfg["k_fold"].get('include_test_split', False)
        self.k_i = cfg["k_fold"].get("k_i", 1)
        self.k_fold_dataset_indices = cfg["k_fold"].get("k_fold_dataset_indices", None)
        if self.k_fold_dataset_indices is None:
            self.k_fold_dataset_indices = self.get_k_fold_dataset_indices()
        self.ensemble_eval = cfg["k_fold"].get("ensemble_eval", False)
        self._check_ensemble_eval_mode()
        # self.hold_out_test_set = cfg["k_fold"].get('hold_out_test_set', False)
        # self.test_set_ratio = cfg["k_fold"].get('test_set_ratio', 0.2)
        self.is_resumed_from_state = is_resumed_from_state
        # VB Removed 20.2.2025: self.k_fold_dataset_indices = None
        self.trained_model_paths = self.cfg["k_fold"].get("trained_model_paths", list())

        log_path = self.save_pb.clone().add('k_fold_log.txt').build()
        self.logger = create_custom_logger(self.k_fold_name, log_path)
        if self.is_resumed_from_state:
            self.logger.info(f"Resuming {self.k}-Fold cross-validation at iteration: {self.k_i} of {self.k}")
            if len(self.trained_model_paths) > 0 and self.ensemble_eval is not False:
                    self._check_previous_checkpoints()

    @beartype
    def _check_ensemble_eval_mode(self):
        assert self.ensemble_eval is False or self.ensemble_eval == "on_best_checkpoints" or \
               self.ensemble_eval == "on_last_checkpoints", "ensemble_eval setting must be either False, " \
                                                            "'on_best_checkpoints' or 'on_last_checkpoints'"
        checkpoints_cfg = self.cfg["settings"]["checkpoints"]
        if self.ensemble_eval == "on_best_checkpoints":
            is_compatible = checkpoints_cfg["save_mode"] == "best" or checkpoints_cfg["save_mode"] == "best_and_last"
            assert is_compatible, "'ensemble_eval' is set to 'on_best_checkpoints', but 'checkpoints -> mode' setting " \
                                  "is not set to 'best' or 'best_and_last'"
        elif self.ensemble_eval == "on_last_checkpoints":
            is_compatible = checkpoints_cfg["save_mode"] == "last" or checkpoints_cfg["save_mode"] == "best_and_last"
            assert is_compatible, "'ensemble_eval' is set to 'on_last_checkpoints', but 'checkpoints -> mode' setting " \
                                  "is not set to 'last' or 'best_and_last'"

        if self.ensemble_eval is not False:
            final_eval_epochs = self.cfg["settings"].get("final_eval_epochs", 1)
            err_msg = f"'ensemble_eval' is set to {self.ensemble_eval}, but 'final_eval_epochs' is {final_eval_epochs}." \
                      f" This is not supported for ensemble evaluation."
            assert final_eval_epochs == 1, err_msg

    @beartype
    def _get_k_fold_folder_name(self):
        model_name = self.cfg["architecture"].get("model_name", "").lower()
        return f"{model_name}_{get_current_date_time_str()}"

    @beartype
    def _check_previous_checkpoints(self):
        checkpoints_not_found = []
        for checkpoint_path in self.trained_model_paths:
            if not os.path.exists(checkpoint_path):
                checkpoints_not_found.append(checkpoint_path)
        if len(checkpoints_not_found) > 0:
            not_found_list = "\n".join(checkpoints_not_found)
            raise ValueError(f"{self.k}-Fold cross-validation was resumed and do_ensemble_eval is True, "
                             f"but the following previously trained checkpoints were not found: "
                             f"{not_found_list}")

    @beartype
    def _log_iteration(self):
        self.logger.info(f"Starting {self.k}-Fold cross-validation iteration {self.k_i} of {self.k}")

    @beartype
    def _set_k_i(self, k_i: int):
        self.k_i = k_i
        self.cfg["k_fold"]["k_i"] = k_i

    @beartype
    def _set_k_fold_dataset_indices(self):
        self.k_fold_dataset_indices = self.get_k_fold_dataset_indices()
        self.cfg["k_fold"]["k_fold_dataset_indices"] = self.k_fold_dataset_indices

    @beartype
    def run(self):
        if self.k_fold_dataset_indices is None:
            self._set_k_fold_dataset_indices()
        assert len(self.k_fold_dataset_indices) == self.k, "Number of folds are not equal to k"
        print(f"Results will be stored in {self.save_pb.clone().build()}")
        self._log_iteration()

        for i in range(self.k_i, self.k + 1):  # Start counting with value 1
            self._set_k_i(i)
            self.save_state()  # important that this is before training, but after setting k_i

            # Check if there has already been started a trial for the current fold.
            # If yes, continue from the last_checkpoint.pt, if not start the fold from scratch with a new trial

            # Search for the correct fold directory
            potential_ckpt_path = ""
            fold_dir = os.path.join(self.save_pb.clone().dataset().build())
            fold_pattern = os.path.join(fold_dir, f"fold_{i}_{self.k_fold_name.split('_')[0]}*")  # Wildcard * for timestamp
            matching_folders = glob.glob(fold_pattern)

            if len(matching_folders) > 0:
                # If more than one folder is found, raise a warning
                if len(matching_folders) > 1:
                    self.logger.warning(f"Found more than one previous trial for fold {i}. "
                                        f"Continuing with the first match.")
                # Take the first match (assuming only one folder per fold exists)
                fold_dir = matching_folders[0]
                potential_ckpt_path = os.path.join(fold_dir, "latest_checkpoint.pt")
            else:
                self.logger.info(f"No previous trial found for fold {i}.")

            if self.is_resumed_from_state and os.path.exists(potential_ckpt_path):
                # The checkpoint already contains the trial_name and the corresponding train, valid and test indices
                checkpoint = torch.load(potential_ckpt_path)
                cur_epochs = checkpoint['current_epoch']
                cur_iters = checkpoint['current_iteration']
                print(f"Checkpoint was stopped after {cur_epochs} epochs (={cur_iters} iterations).")
                trainer = Trainer.from_state_dict(checkpoint, self.cfg_path)
                self._check_dataset_indices(trainer.state.dataset_manager, self.k_fold_dataset_indices[i - 1])
                # Manually trigger the scheduler update if the scheduler is not None (for learning rate adaptation)
                if trainer.state.scheduler is not None:
                    trainer.update_scheduler()
                self.logger.info(f"Resuming fold {i} from last checkpoint in epoch {cur_epochs}...")
                trainer.train()
                trial_out_path = trainer.trial_path_builder.clone().build()
                self._add_checkpoint_path(trial_out_path)
                continue
            else:
                trial_cfg = deepcopy(self.cfg)
                model_name = trial_cfg["architecture"].get("model_name", "").lower()
                self.logger.info(f"Starting fold {i} from scratch...")
                trial_cfg['trial_name'] = f"fold_{i}_{model_name}_{get_current_date_time_str()}"
                trial_cfg["dataset"][f"{SplitType.TRAIN.value.lower()}_indices"] = self.k_fold_dataset_indices[i - 1][0]
                trial_cfg["dataset"][f"{SplitType.VAL.value.lower()}_indices"] = self.k_fold_dataset_indices[i - 1][1]
                trial_cfg["dataset"][f"{SplitType.TEST.value.lower()}_indices"] = self.k_fold_dataset_indices[i - 1][2]
                trainer = Trainer(trial_cfg, self.cfg_path)
                self._check_dataset_indices(trainer.state.dataset_manager, self.k_fold_dataset_indices[i - 1])
                trainer.train()
                trial_out_path = trainer.trial_path_builder.clone().build()
                self._add_checkpoint_path(trial_out_path)

        self.save_state()
        self.logger.info(f"{self.k}-fold cross-validation finished.")
        if self.ensemble_eval is not False:
            self.logger.info(f"Running ensemble evaluation {self.ensemble_eval.replace('_', ' ')}...")
            self.run_ensemble_eval()

    @beartype
    def _check_dataset_indices(self, ds_manager: DatasetManager, k_ds_indices: Tuple[List[int], List[int], List[int]]):
        supposed_indices = {
            SplitType.TRAIN: set(k_ds_indices[0]),
            SplitType.VAL: set(k_ds_indices[1]),
            SplitType.TEST: set(k_ds_indices[2])
        }
        for split_type, indices in supposed_indices.items():
            ds = ds_manager.get_dataset(split_type)
            if ds is None:
                assert len(indices) == 0
            else:
                assert indices == set(
                    ds.indices), f"Dataset indices for {split_type.value} are not equal to supposed indices"

    @beartype
    def _add_checkpoint_path(self, trial_out_path):
        if self.ensemble_eval is not False:
            if self.ensemble_eval == "on_best_checkpoints":
                checkpoint_filenames = ['best_checkpoint.pt']
            elif self.ensemble_eval == 'on_last_checkpoints':
                checkpoint_filenames = ['last_checkpoint.pt']
            else:
                raise ValueError(f"ensemble_eval setting is {self.ensemble_eval}, but it must be either "
                                 f"'on_best_checkpoints' or 'on_last_checkpoints' or False")
            checkpoint_path = find_file(trial_out_path, checkpoint_filenames)
            assert checkpoint_path is not None, f"Could not find checkpoint file in {trial_out_path}"
            self.trained_model_paths.append(checkpoint_path)
            self.cfg["k_fold"]["trained_model_paths"] = self.trained_model_paths

    @beartype
    def run_ensemble_eval(self):
        assert len(self.trained_model_paths) > 0, "No trained models found"
        main_eval_split = SplitType.TEST if self.include_test_split else SplitType.VAL
        ensemble_evaluator = EnsembleEvaluator(self.trained_model_paths, self.save_pb.clone(), main_eval_split,
                                               self.logger)
        ensemble_evaluator.run_evaluation()

    @classmethod
    @beartype
    def from_state_dict(cls, state_dict: dict, cfg_or_dir_path: Optional[str]) -> 'KFoldCrossValidator':
        cross_validator = cls(state_dict['cfg'], cfg_or_dir_path, is_resumed_from_state=True)
        return cross_validator

    @beartype
    def save_state(self) -> None:
        state = {
            'cfg': self.cfg,
        }
        file_path = self.save_pb.clone().add('k_fold_state.pkl').build()
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    @beartype
    def get_k_fold_dataset_indices(self) -> List[Tuple[List[int], List[int], List[int]]]:
        dataset = DatasetManager.build_dataset(self.cfg, SplitType.ALL)
        all_sample_indices = dataset.indices
        assert all_sample_indices is not None and len(all_sample_indices) > 0, "Dataset must have indices"

        folds: List[List[int]] = get_folds(all_sample_indices, self.k, self.random_seed)

        assert len(folds) == self.k, "Number of folds are not equal to k"
        assert sum([len(fold) for fold in folds]) == len(all_sample_indices), "Folds do not cover all indices"
        all_indices_set = set()
        for fold_samples in folds:
            fold_samples_set = set(fold_samples)
            assert len(all_indices_set.intersection(fold_samples_set)) == 0, "Folds are overlapping"
            all_indices_set.update(fold_samples_set)

        folds_indices = {i for i in range(self.k)}  # set of {0, 1, ..., k-1}
        k_fold_dataset_indices: List[Tuple[List[int], List[int], List[int]]] = []
        for i_k in range(self.k):

            val_fold_idx = i_k % self.k
            val_sample_indices = folds[val_fold_idx]

            test_fold_idx = (i_k + 1) % self.k
            test_sample_indices = folds[test_fold_idx] if self.include_test_split else []

            train_fold_idxs = folds_indices.copy()
            train_fold_idxs.remove(val_fold_idx)
            if self.include_test_split:
                train_fold_idxs.remove(test_fold_idx)
            train_fold_idxs = sorted([idx for idx in train_fold_idxs])
            train_sample_indices = [sample_i for i in train_fold_idxs for sample_i in folds[i]]
            k_fold_dataset_indices.append((train_sample_indices, val_sample_indices, test_sample_indices))

        return k_fold_dataset_indices
