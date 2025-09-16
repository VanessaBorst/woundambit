# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import gc
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from medseg.config.config import merge_configs
from medseg.data.dataset_manager import DatasetManager
from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType
from medseg.evaluation.loss_tracker import LossTracker
from medseg.evaluation.metrics import EvalMetric
from medseg.evaluation.metrics_manager import MetricsManager
from medseg.evaluation.params import get_model_args
from medseg.models.model_builder import build_model
from medseg.training.early_stop import EarlyStop, get_early_stop
from medseg.training.loss.loss_builder import get_loss_module
from medseg.training.optimizers import get_optimizer
from medseg.training.schedulers import build_scheduler_and_metric
from medseg.util.date_time import get_current_date_time_str
from medseg.util.logger import create_custom_logger
from medseg.util.path_builder import PathBuilder
from medseg.util.random import ensure_reproducibility


@dataclass
class TrainerState:
    """
    A class representing the state of a trainer, including the configuration, model,
    device, optimizer, scheduler, scaler, and other related components.

    Attributes:
        cfg (dict): Configuration dictionary.
        model (Any): Model instance.
        device (Any): Device to run the model on.
        optimizer (Any): Optimizer for the model.
        scheduler (Any): Learning rate scheduler.
        scheduler_metric (EvalMetric): The evaluation metric used by the scheduler.
        scaler (Any): Gradient scaler for mixed-precision training.
        early_stop (EarlyStop): Early stopping mechanism.
        current_epoch (int): Current epoch number.
        current_iteration (int): Current iteration number.
        max_epochs (int): Maximum number of epochs.
        max_iterations (int): Maximum number of iterations.
        datasets (Dict[SplitType, MedsegDataset]): Datasets for training, validation, and testing.
        loaders (Dict[SplitType, DataLoader]): DataLoaders for training, validation, and testing.
        multiclass (bool): Whether the dataset is multiclass or not.
        train_iteration_func (Callable): Function for a single training iteration.
        loss_func (Callable): Function for computing the loss.
        train_loss_tracker (LossTracker): Loss tracking utility.
        mixed_precision (bool): Whether to use mixed-precision training or not.
        metrics_manager (MetricsManager): Metrics management utility.
        tbx_writer (SummaryWriter): TensorBoardX writer.
        is_hyperopt_trial (bool): Whether the trainer is in a hyperparameter tuning run.
        full_determinism (bool): Whether to enforce full determinism.

    Methods:
        state_dict(): Returns the state dictionary of the TrainerState.
        load_state_dict(state_dict): Initializes a TrainerState from a state dictionary.
    """

    def __init__(self, cfg: dict):
        """
        Initializes a TrainerState from a configuration dictionary.
        Args:
           cfg (dict): The configuration dictionary to initialize the TrainerState.
        """
        self.cfg = deepcopy(cfg)

        # TODO: Maybe this can be merged and unified for both
        hyperopt_cfg = self.cfg.get('trial_param_space', None)
        if hyperopt_cfg is not None and self.cfg["hyperopt"]["type"]=="hyperband":
            # This is used to merge the param dicts produced by Hyperband
            self.cfg = merge_configs(hyperopt_cfg, self.cfg)
            self.is_hyperopt_run = True
        else:
            # In case of grid_search, the params are already updated
            self.is_hyperopt_run = False if self.cfg.get('hyperopt', None) is None else True

        k_fold_cfg = self.cfg.get('k_fold', None)
        self.is_k_fold_run = k_fold_cfg is not None

        if self.is_k_fold_run and self.is_hyperopt_run:
            raise ValueError("Both k-fold cross-validation and hyperparameter tuning are enabled")

        if 'trial_name' not in self.cfg:
            self.cfg['trial_name'] = f"{self.cfg['architecture']['model_name']}_{get_current_date_time_str()}"
        self.trial_name = self.cfg['trial_name']
        log_path = PathBuilder.trial_out_builder(self.cfg).add('training_log.txt').build()
        self.logger = create_custom_logger(f"logger_{self.trial_name}", log_path)
        ensure_reproducibility(self.cfg['settings']['random_seed'])
        dataset_manager = DatasetManager(self.cfg)
        class_mapping = dataset_manager.get_train_dataset().class_mapping
        out_channels = class_mapping.num_classes if class_mapping.multiclass else 1
        torch.hub.set_dir(PathBuilder.pretrained_dir_builder().build())  # set pretrained weights download dir
        model = build_model(self.cfg, out_channels=out_channels)
        model_args = model.init_args if model.init_args is not None else {}
        model_args.update(get_model_args(self.cfg))
        optim = get_optimizer(self.cfg, model.parameters())
        model.set_param_groups(optim)
        scheduler, scheduler_metric = build_scheduler_and_metric(optim, cfg)
        early_stop = get_early_stop(self.cfg)
        train_iteration_func = model.train_iteration
        multiclass = dataset_manager.get_train_dataset().is_multiclass()
        loss_from_cfg = get_loss_module(self.cfg)
        loss_func = model.default_loss_func(multiclass) if loss_from_cfg is None else loss_from_cfg
        tbx_writer = self._get_tbx_writer()

        # Set attributes
        self.hyperopt_cfg = hyperopt_cfg
        self.model = model
        self.model_args = model_args
        self.compiled_model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim
        self.scheduler = scheduler
        self.scheduler_metric = scheduler_metric
        self.scaler = GradScaler()
        self.early_stop = early_stop
        self.current_epoch = 0
        self.current_iteration = 0
        self.max_epochs = self.cfg['settings'].get('max_epochs', None)
        self.max_iterations = self.cfg['settings'].get('max_iterations', None)
        self.dataset_manager = dataset_manager
        self.multiclass = multiclass
        self.train_iteration_func = train_iteration_func
        self.loss_func = loss_func
        self.mixed_precision = self.cfg['settings']['mixed_precision']
        self.train_loss_tracker = LossTracker(SplitType.TRAIN, tbx_writer)
        self.metrics_manager = MetricsManager(self.cfg, class_mapping, self.trial_name, tbx_writer)
        self.tbx_writer = tbx_writer
        self.full_determinism = self.cfg['settings']['full_determinism']
        self.is_from_checkpoint = False
        self.save_sample_segmentations = self.cfg['settings'].get('save_sample_segmentations', False)
        self.eval_object_sizes = self.cfg['settings'].get('eval_object_sizes', False)

    def state_dict(self):
        """
        Returns the state dictionary of the TrainerState.

        Returns:
           dict: A dictionary containing the state of the TrainerState.
       """
        return {
            'cfg': self.cfg,
            'hyperopt_cfg': self.hyperopt_cfg,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler': self.scaler.state_dict(),
            'current_epoch': self.current_epoch,
            'current_iteration': self.current_iteration,
            'train_loss_tracker': self.train_loss_tracker.state_dict(),
            'early_stop': self.early_stop.state_dict() if self.early_stop is not None else None,
            'metrics_manager': self.metrics_manager.state_dict(),
        }

    def load_state_dict(self, state_dict: dict, new_trial_name: Optional[str] = None):
        """
        Initializes a TrainerState from a state dictionary.

        Args:
            state_dict (dict): The state dictionary to initialize the TrainerState.
        """
        self.cfg = state_dict['cfg']
        self.hyperopt_cfg = state_dict['hyperopt_cfg'] if 'hyperopt_cfg' in state_dict else None
        self.model.to(self.device)  # has to be done here, otherwise optimizer will cause exceptions later
        self.model.load_state_dict(state_dict['model'])
        self.compiled_model = self.model
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.model.set_param_groups(self.optimizer)
        if state_dict['scheduler'] is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.current_epoch = state_dict['current_epoch']
        self.current_iteration = state_dict['current_iteration']
        if state_dict['early_stop'] is not None:
            self.early_stop.load_state_dict(state_dict['early_stop'])
        self.train_loss_tracker.load_state_dict(state_dict['train_loss_tracker'])
        self.metrics_manager.load_state_dict(state_dict['metrics_manager'])
        self.trial_name = state_dict['cfg']['trial_name'] if new_trial_name is None else new_trial_name
        self.is_hyperopt_run = self.hyperopt_cfg is not None
        self.is_from_checkpoint = True
        self.early_stop.load_state_dict(state_dict['early_stop'])

        self.tbx_writer.close()
        self.tbx_writer = self._get_tbx_writer()
        self.metrics_manager.tbx_writer = self.tbx_writer
        self.train_loss_tracker.tbx_writer = self.tbx_writer

    def free_memory(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.early_stop = None
        self.dataset_manager = None
        self.metrics_manager = None
        self.train_loss_tracker = None
        self.tbx_writer = None
        torch.cuda.empty_cache()
        gc.collect()

    def _get_tbx_writer(self):
        return SummaryWriter(log_dir=PathBuilder.tb_out_builder(self.cfg).build())
