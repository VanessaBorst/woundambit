# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import json
import os
import shutil
import sys
import time
from typing import Optional, List
from calflops import calculate_flops

import torch
from tqdm import tqdm

from medseg.config.parse import parse_checkpoint_cfg
from medseg.data.split_type import SplitType
from medseg.evaluation.evaluator import Evaluator
from medseg.evaluation.params import get_total_params, create_model_summary
from medseg.training.trainer_state import TrainerState
from medseg.util.date_time import get_current_date_time_str
from medseg.util.files import save_text_to_file
from medseg.util.logger import flatten_dict
from medseg.util.path_builder import PathBuilder
from medseg.util.random import use_deterministic


class Trainer:
    """
    Trainer class responsible for training a medical segmentation model.

    Attributes:
        state (TrainerState): An object containing the trainer state and related information.
    """

    def __init__(self,
                 cfg: dict,
                 cfg_path: str = None):
        """
        Initialize the Trainer class with a given configuration.

        Args:
            cfg (dict): Configuration dictionary containing model and training parameters.
        """
        cfg = parse_checkpoint_cfg(cfg)
        self.state = TrainerState(cfg)
        self.trial_path_builder = PathBuilder.trial_out_builder(cfg)
        if cfg_path is not None:
            config_file_name = os.path.basename(cfg_path)
            shutil.copy(cfg_path, self.trial_path_builder.clone().add(config_file_name).build())
            self.state.tbx_writer.add_text('Config file', config_file_name, 0)
        self.checkpoints_cfg = cfg['settings']['checkpoints']
        self.best_checkpoint_metric_value = None
        self.evaluator_val = None
        self.evaluator_test = None
        self.evaluator_final = None
        self.is_val_available = self.state.dataset_manager.has_split(SplitType.VAL)
        self.is_test_available = self.state.dataset_manager.has_split(SplitType.TEST)
        self.final_eval_epochs = cfg['settings'].get('final_eval_epochs', 1)
        self.last_epoch_with_val_evaluation = -1
        if self.state.is_hyperopt_run and not self.is_val_available:
            self.log("Running hyperparameter optimization without a validation split in the dataset. If th"
                     "is is unintended, check the dataset configuration and run again. Proceeding... ")

    @classmethod
    def from_state_dict(cls, state_dict: dict, new_trial_name: Optional[str] = None):
        """
        Create a Trainer instance from a state dictionary.

        Args:
            state_dict (dict): The state dictionary containing the trainer's configuration and state.
            new_trial_name (str, None): The name of the new trial. If None, the trial name from the state dictionary is used.
        Returns:
            Trainer: A new Trainer instance.
        """
        trainer = cls(state_dict['cfg'])
        trainer.state.load_state_dict(state_dict, new_trial_name)
        if state_dict.get("early_stop") and state_dict["early_stop"].get("best_checkpoint_metric_value"):
            trainer.best_checkpoint_metric_value = state_dict["early_stop"]["best_checkpoint_metric_value"]
            print(f"Loaded best checkpoint metric value: {trainer.best_checkpoint_metric_value}")
        return trainer

    def prepare_for_training(self):
        """
        Prepare the model and the environment for training.
        """

        save_text_to_file(self.state.model_args.__str__(),
                          self.trial_path_builder.clone().add("model_args.txt").build())
        started_str = "Resumed" if self.state.is_from_checkpoint else "Initialized"
        self.state.tbx_writer.add_text(
            tag=f"{SplitType.TRAIN.value}/info",
            text_string=f"{started_str} training {get_current_date_time_str()} on dataset "
                        f"{self.state.dataset_manager.get_train_dataset().__class__.__name__} with model "
                        f"{self.state.cfg['architecture']['model_name']} "
                        f"with {get_total_params(self.state.model)} learnable parameters.",
            global_step=self.state.current_epoch
        )

        if self.state.full_determinism: use_deterministic()

        self.state.model.to(self.state.device)
        epochs_to_train = self.state.max_epochs + 1 - self.state.current_epoch
        if self.state.cfg['settings'].get('torch_compile', True) and epochs_to_train > 10:
            # when this is not set, torch prints out warning during training on
            # A100 MiG partition (unsure if this is the case for other setups)
            torch.set_float32_matmul_precision('high')
            self.state.compiled_model = torch.compile(self.state.model, mode='default')

        final_eval_split = self.get_split_for_final_eval()
        save_samples_val = self.state.save_sample_segmentations and final_eval_split == SplitType.VAL
        save_samples_test = self.state.save_sample_segmentations and final_eval_split == SplitType.TEST
        eval_object_sizes_val = self.state.eval_object_sizes and final_eval_split == SplitType.VAL
        eval_object_sizes_test = self.state.eval_object_sizes and final_eval_split == SplitType.TEST
        self.evaluator_val = Evaluator.from_trainer_state(self.state,
                                                          SplitType.VAL,
                                                          save_samples_val,
                                                          eval_object_sizes_val)
        self.evaluator_test = Evaluator.from_trainer_state(self.state,
                                                           SplitType.TEST,
                                                           save_samples_test,
                                                           eval_object_sizes_test)
        self.log(get_training_info(self.state))
        if self.state.is_from_checkpoint:
            self.log(f"Resuming model training from epoch {self.state.current_epoch}")
        else:
            self.log("Starting model training...")

    def train(self):
        """
        Train the model and perform evaluation on validation and test datasets.
        """
        self.prepare_for_training()
        epoch_durations = []
        eval_durations = []

        for epoch in range(self.state.current_epoch, self.state.max_epochs):
            self.state.current_epoch += 1

            # Train epoch
            self.log(f"Epoch {self.state.current_epoch} of {self.state.max_epochs}...")
            epoch_start = time.time()
            self.train_epoch()
            epoch_duration = time.time() - epoch_start
            epoch_durations.append(epoch_duration)
            self.log(f"Epoch {self.state.current_epoch} took {epoch_duration:.2f} seconds.")

            if self.is_val_available:
                # Perform evaluation
                eval_start = time.time()
                self.evaluator_val.evaluate(self.state.current_epoch)
                eval_duration = time.time() - eval_start
                eval_durations.append(eval_duration)
                self.log(f"Evaluation on validation set took {eval_duration:.2f} seconds.")
                self.last_epoch_with_val_evaluation = self.state.current_epoch
                self.save_metrics_txt_and_checkpoint_if_allowed(SplitType.VAL)

            # Save plots
            metrics_to_plot = self.state.metrics_manager.tracked_metrics + ['mean_loss']
            self.state.metrics_manager.save_plots(SplitType.TRAIN, metrics_to_plot)

            if self.is_val_available:
                self.state.metrics_manager.save_plots(SplitType.VAL, metrics_to_plot)

            # Save checkpoint, determine early stop, update scheduler

            if self.should_stop_early():
                self.log(f"Early stop triggered in epoch {self.state.current_epoch}. Stopping training...")
                break
            self.update_scheduler()

        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations) if len(epoch_durations) > 0 else 0
        avg_eval_duration = sum(eval_durations) / len(eval_durations) if len(eval_durations) > 0 else 0
        avg_duration = avg_epoch_duration + avg_eval_duration
        self.log(f"######################## FINISHED TRAINING #####################################")
        self.log(f"Average time per epoch (including evaluation): {avg_duration:.2f} seconds.")
        # Perform final evaluation, metrics saving, tensorboard updates etc
        self.finalize_training()

    def train_epoch(self):
        """
        Train the model for a single epoch.
        """

        self.state.model.train()

        self.state.train_loss_tracker.reset()
        metrics_tracker = self.state.metrics_manager.add_tracker(SplitType.TRAIN)

        loop = tqdm(self.state.dataset_manager.get_train_loader(), file=sys.stderr)
        for batch_idx, ([images, masks], ids) in enumerate(loop):
            self.state.train_iteration_func(images, masks, ids, self.state)
            loop.set_postfix(loss=self.state.train_loss_tracker.get_last())

        # Compute and record metrics
        metrics_tracker.compute_total_metrics()
        metrics_tracker.sync_to_tensorboard(self.state.current_epoch)
        mean_loss_train = self.state.train_loss_tracker.compute_mean()
        self.state.metrics_manager.add_mean_loss(mean_loss_train, SplitType.TRAIN, self.state.current_epoch)
        self.state.train_loss_tracker.sync_to_tensorboard()
        current_lr = self.state.optimizer.param_groups[0]['lr']
        self.state.tbx_writer.add_scalar(f"{SplitType.TRAIN.value}/lr", current_lr, self.state.current_epoch)

    def finalize_training(self):
        """
        Finalize the training process by performing evaluation on the test or val dataset and saving metrics.
        """

        self.log(f"Model training finished (Trial {self.state.cfg['trial_name']})!")

        split = self.get_split_for_final_eval()
        do_val_eval = self.last_epoch_with_val_evaluation != self.state.current_epoch
        do_val_eval = do_val_eval or self.final_eval_epochs > 1

        if split == SplitType.TEST:
            self.log(
                f"Performing final evaluation on test dataset(s) for the current epoch {self.state.current_epoch}...")
            self.evaluator_test.evaluate(self.state.current_epoch, for_epochs=self.final_eval_epochs,
                                         is_final_eval=True)
            self.save_metrics_txt_and_checkpoint_if_allowed(split, is_final_checkpoint=True)

        elif split == SplitType.VAL and do_val_eval:
            self.log("Performing final evaluation on validation dataset...")
            self.evaluator_val.evaluate(self.state.current_epoch, for_epochs=self.final_eval_epochs, is_final_eval=True)
            self.save_metrics_txt_and_checkpoint_if_allowed(split, is_final_checkpoint=True)

        self.log("Saving metrics output...")
        self.state.metrics_manager.sync_hparams_to_tensorboard(self.get_hparams(), split, self.state.current_epoch)
        self.state.metrics_manager.save_plots(split)
        self.state.metrics_manager.save_full_metrics()
        if not self.state.is_hyperopt_run: self.save_model_summary()
        self.state.tbx_writer.close()

    def save_model_summary(self):
        img_size = self.state.cfg['architecture']['in_size']
        channels = self.state.dataset_manager.get_train_dataset().img_channels
        input_size = (1, channels, img_size, img_size)
        model_summary_path = self.trial_path_builder.clone().add('model_summary.txt').build()
        create_model_summary(self.state.model, input_size, self.state.device, model_summary_path)

    def get_hparams(self):
        hparams = {'learnable_params': get_total_params(self.state.model), 'trial_name': self.state.cfg['trial_name']}
        if self.state.is_hyperopt_run:
            hyperopt_cfg_readable = json.dumps(self.state.hyperopt_cfg, indent=2)
            hyperopt_tag = "Parameter permutation from the Hyperoptimization run"
            self.state.tbx_writer.add_text(hyperopt_tag, hyperopt_cfg_readable, global_step=self.state.current_epoch)
            # only add end node values from trial_params (i.e. no nested dicts)
            hyperopt_trial_params = flatten_dict(self.state.hyperopt_cfg, sep=' -> ')
            hparams.update(hyperopt_trial_params)
        else:
            bs_and_lr = {
                "batch_size": self.state.cfg["settings"]["batch_size"],
                "learning_rate": self.state.cfg["optimizer"]["lr"],
            }
            hparams.update(bs_and_lr)
            hparams.update(self.state.model_args)
        # convert unsupported types to string
        hparams = {k: str(v) if isinstance(v, (list, tuple, dict)) else v for k, v in hparams.items()}
        return hparams

    def free_memory(self):
        """
        Free memory
        """
        self.evaluator_test = None
        self.evaluator_val = None
        self.state.free_memory()

    def save_metrics_txt_and_checkpoint_if_allowed(self, split: SplitType, is_final_checkpoint=False):
        """
        Save the current trainer state to a checkpoint file.

        This should be executed after the evaluation on the validation dataset to save the correct metrics.

        Args:
            is_final_checkpoint (bool, optional): If True, indicates that this is the final checkpoint of the training process.
        """

        filenames = self.get_checkpoint_filenames_if_save_allowed()

        for filename in filenames:
            self.log(f"Saving checkpoint {filename} and last {split.get_full_name().lower()} metrics...")
            torch.save(self.state.state_dict(), self.trial_path_builder.clone().add(f"{filename}.pt").build())
            self.state.metrics_manager.save_metrics_txt(split, filename, self.state.current_epoch)

        if is_final_checkpoint:
            # save final metrics in any case, but not necessarily the checkpoint
            split_final = self.get_split_for_final_eval()
            self.state.metrics_manager.save_metrics_txt(split_final, 'final', self.state.current_epoch)


    def get_split_for_final_eval(self) -> SplitType:
        if self.state.is_hyperopt_run:
            split = SplitType.VAL if self.is_val_available else SplitType.TRAIN
        elif self.state.dataset_manager.has_test_or_aux_test_split():
            split = SplitType.TEST
        elif self.is_val_available:
            split = SplitType.VAL
        else:
            split = SplitType.TRAIN
        return split

    def get_checkpoint_filenames_if_save_allowed(self) -> List[str]:
        """
        Determine if the current checkpoint should be saved. Return empty list if not and a list of filenames otherwise.
        Returns:
            bool: True if the checkpoint should be saved, False otherwise.
            str: The checkpoint name and also the reason for saving the checkpoint.
        """
        filename_list = []
        if self.checkpoints_cfg['save_mode'] == 'none':
            return []
        elif self.state.current_epoch >= self.checkpoints_cfg['min_epoch']:
            if self.checkpoints_cfg['save_mode'] == 'best' or self.checkpoints_cfg['save_mode'] == 'best_and_last':
                split = SplitType.VAL if self.is_val_available else SplitType.TRAIN
                metric_value = self.state.metrics_manager.get_last_metric(split, self.checkpoints_cfg['metric'])
                if self.best_checkpoint_metric_value is None or metric_value > self.best_checkpoint_metric_value:
                    self.best_checkpoint_metric_value = metric_value
                    filename_list.append('best_checkpoint')
            if self.checkpoints_cfg['save_mode'] == 'last' or self.checkpoints_cfg['save_mode'] == 'best_and_last':
                filename_list.append('latest_checkpoint')
        return filename_list

    def should_stop_early(self) -> bool:
        """
        Check if early stopping criteria are met.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        if self.state.early_stop is not None and self.state.current_epoch > 1:
            split = SplitType.VAL if self.is_val_available else SplitType.TRAIN
            self.state.early_stop.check_metric(self.state.metrics_manager.metrics_trackers[split])
            if self.state.early_stop.counter > 0:
                self.log(f"No (large enough) improvements for {self.state.early_stop.counter} epochs")
            return self.state.early_stop.stop_triggered
        return False

    def update_scheduler(self):
        """
        Update the learning rate scheduler based on the latest metric from val or train set.
        """
        scheduler = self.state.scheduler
        if scheduler is not None:
            metric_key = self.state.scheduler_metric
            if metric_key is not None:
                split = SplitType.VAL if self.is_val_available else SplitType.TRAIN
                last_scheduler_metric = self.state.metrics_manager.get_last_metric(split, metric_key)
                scheduler.step(last_scheduler_metric)
            else:
                scheduler.step()

    def log(self, msg: str):
        self.state.logger.info(msg)


def get_training_info(state: TrainerState) -> str:
    ds_name = state.cfg['dataset']['type'] if 'type' in state.cfg['dataset'] else state.cfg['dataset']['path']
    trial_out_folder = PathBuilder.trial_out_builder(state.cfg).build()
    ds_train = state.dataset_manager.get_train_dataset()
    ds_val = state.dataset_manager.get_val_dataset()
    ds_test = state.dataset_manager.get_test_dataset()
    train_size = len(ds_train) if ds_train is not None else 0
    val_size = len(ds_val) if ds_val is not None else 0
    test_size = len(ds_test) if ds_test is not None else 0
    total_size = train_size + val_size + test_size
    train_ratio = train_size / total_size
    val_ratio = val_size / total_size
    test_ratio = test_size / total_size
    train_percent = round(train_ratio * 100)
    val_percent = round(val_ratio * 100)
    test_percent = round(test_ratio * 100)

    flops, macs, _ = calculate_flops(model=state.model,
                                     input_shape=(1,
                                                  state.dataset_manager.get_train_dataset().img_channels,
                                                  state.model_args["in_size"], state.model_args["in_size"]),
                                     output_precision=4,
                                     print_results=False,
                                     print_detailed=False)

    model_args_str = ''
    for k, v in state.model_args.items():
        model_args_str += f"\n                {k}: {str(v)}"

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "Not Set")

    training_info = f"""
    ============================================
        Starting training with the following settings:
        Model:
            Architecture: {state.cfg['architecture']['arch_type']}
            Model name: {state.cfg['architecture']['model_name']}
            Model args: {model_args_str}
            Learnable Parameters: {get_total_params(state.model)}
            MACs: {macs}
            FLOPs: {flops}
        Dataset:
            Name: {ds_name}
            Total size: {total_size}
            Train size: {train_size} (~{train_percent}%)
            Validation size: {val_size} (~{val_percent}%)
            Test size: {test_size} (~{test_percent}%)
        Training settings:
            Maximum epochs: {state.max_epochs}
            Batch size: {state.cfg['settings']['batch_size']}
            Learning rate: {state.optimizer.param_groups[0]['lr']}
            Device: {state.device}  (CUDA_VISIBLE_DEVICES: {cuda_visible_devices})
            Output folder for trial: {trial_out_folder}
    ============================================
    """
    return training_info
