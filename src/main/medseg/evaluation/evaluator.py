import os
from logging import Logger
from typing import Callable, Optional, List, Dict

import torch
from beartype import beartype
from calflops import calculate_flops
from tensorboardX import SummaryWriter
from torch import nn, Tensor
from torch.utils.data import DataLoader

from medseg.data.dataset_manager import DatasetManager
from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType
from medseg.evaluation.loss_tracker import LossTracker
from medseg.evaluation.medcam import medcam
from medseg.evaluation.metrics_manager import MetricsManager
from medseg.evaluation.segmentation_visualizer import SegmentationVisualizer, ImageSaveMode
from medseg.models.model_builder import build_model
from medseg.training.loss.loss_builder import get_loss_module
from medseg.training.trainer_state import TrainerState
from medseg.util.date_time import get_current_date_time_str
from medseg.util.img_ops import logits_to_segmentation_mask
from medseg.util.logger import create_custom_logger
from medseg.util.path_builder import PathBuilder
from medseg.util.random import ensure_reproducibility


class Evaluator:
    """
    Class for evaluating a model on a given dataset split.
    """

    @beartype
    def __init__(self, cfg: dict, model: nn.Module, compiled_model: Callable, split: SplitType,
                 dataset_manager: DatasetManager, metrics_manager: MetricsManager, device: torch.device,
                 loss_func: Callable, logger: Logger, base_pb: PathBuilder, save_sample_segmentations: bool = False,
                 eval_object_sizes: bool = False):
        self.cfg = cfg
        self.model = model
        self.compiled_model = compiled_model
        self.split = split
        self.dataset_manager = dataset_manager
        self.metrics_manager = metrics_manager
        self.device = device
        self.loss_func = loss_func
        self.loss_tracker = LossTracker(self.split, self.metrics_manager.tbx_writer)
        self.logger = logger
        self.save_sample_segmentations = save_sample_segmentations
        self.save_modes = [ImageSaveMode.RANDOM_SUBSET, ImageSaveMode.WORST]
        self.base_pb = base_pb
        self.eval_object_sizes = eval_object_sizes
        self.metrics_manager.set_eval_object_sizes(eval_object_sizes, self.split)

    @classmethod
    @beartype
    def from_trainer_state(cls, state: TrainerState, split: SplitType, save_sample_segmentations: bool = False,
                           eval_object_sizes: bool = False):
        return cls(state.cfg, state.model, state.compiled_model, split, state.dataset_manager, state.metrics_manager,
                   state.device, state.loss_func, state.logger, PathBuilder.trial_out_builder(state.cfg),
                   save_sample_segmentations, eval_object_sizes)

    @classmethod
    @beartype
    def from_checkpoint(cls, checkpoint: dict, path: str, split: SplitType, use_XAI: bool = False):
        path = os.path.dirname(path)  # get path without filename
        base_pb = PathBuilder().add(path)
        cfg = checkpoint['cfg']
        if 'trial_name' not in cfg:
            trial_name = f"{cfg['architecture']['model_name']}_{get_current_date_time_str()}_EVAL"
            cfg['trial_name'] = trial_name
        else:
            trial_name = cfg['trial_name']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset_manager = DatasetManager(cfg)
        class_mapping = None
        for dataset in dataset_manager.datasets.values():
            class_mapping = dataset.class_mapping if class_mapping is None else class_mapping
        assert class_mapping is not None, "No class mapping found in dataset manager"
        out_channels = class_mapping.num_classes if class_mapping.multiclass else 1
        model = build_model(cfg, out_channels=out_channels)
        model.to(device)
        model.load_state_dict(checkpoint['model'])
        model_informations = get_model_info(checkpoint['cfg'], dataset_manager, model, split)
        if use_XAI:
            print("The usage of XAI with m3d-cam is activated")
            model = medcam.inject(model, output_dir=base_pb.clone().add(
                f"images_m3dCam_{dataset_manager.datasets[split].get_name().lower()}_{split.get_full_name().upper()}").build(),
                                  backend='gcam',
                                  save_maps=True)
        print("Successfully loaded model from specified checkpoint.")
        # do not compile model here in any case, as the compilation would take far too long relative to the eval
        compiled_model = model
        log_path = base_pb.clone().add('eval_log.txt').build()
        logger = create_custom_logger(f"logger_{cfg['trial_name']}", log_path)
        logger.info(msg=model_informations)
        ensure_reproducibility(cfg['settings']['random_seed'])

        tbx_writer = SummaryWriter(log_dir=base_pb.clone().build())
        metrics_manager = MetricsManager(cfg, class_mapping, trial_name, tbx_writer, base_pb=base_pb)
        multiclass = class_mapping.multiclass
        loss_from_cfg = get_loss_module(cfg)
        loss_func = model.default_loss_func(multiclass) if loss_from_cfg is None else loss_from_cfg
        save_sample_segmentations = cfg['settings'].get('save_sample_segmentations', False)
        eval_object_sizes = cfg['settings'].get('eval_object_sizes', False)
        return cls(cfg, model, compiled_model, split, dataset_manager, metrics_manager, device, loss_func, logger,
                   base_pb, save_sample_segmentations, eval_object_sizes)

    @beartype
    def evaluate(self, training_epoch: int, for_epochs: Optional[int] = None, is_final_eval: bool = False) -> Dict[str,
                                                                                                                   MetricsManager]:
        """
        Evaluates the model on the calculates the evaluation metrics, and syncs the results to TensorBoard.

        Args:
            training_epoch (int): The current epoch of the training run
            for_epochs (Optional[int]): The number of epochs to evaluate for. If None, evaluation is run once
            is_final_eval (bool): Whether this is the final evaluation of the model.
        """
        metrics_managers = dict()
        if self.dataset_manager.has_split(self.split):
            dataset, loader = self.dataset_manager.get_dataset_and_loader(self.split)
            mm = self.evaluate_dataset(dataset, loader, training_epoch, False, for_epochs, is_final_eval=is_final_eval)
            metrics_managers[dataset.get_name().lower()] = mm

        if self.split == SplitType.TEST and self.dataset_manager.has_aux_test_datasets():
            for dataset, loader in self.dataset_manager.get_aux_test_datasets_and_loaders():
                mm = self.evaluate_dataset(dataset, loader, training_epoch, True, for_epochs,
                                           is_final_eval=is_final_eval)
                mm.save_full_metrics()
                metrics_managers[dataset.get_name().lower()] = mm
        return metrics_managers

    @beartype
    def evaluate_dataset(self,
                         dataset: MedsegDataset,
                         loader: DataLoader,
                         training_epoch: int,
                         is_aux_test_data: bool = False,
                         for_epochs: Optional[int] = None,
                         predictions_hook: Optional[Callable[[str, List[str], Tensor], None]] = None,
                         is_final_eval: bool = False,
                         xai_predictions_hook: Optional[Callable[[str, List[str], Tensor], None]] = None,
                         ) -> MetricsManager:
        """
        Evaluates the model on the specified dataset split, calculates the evaluation metrics, and syncs the results
        to TensorBoard.

        Args:
            dataset (MedsegDataset): The dataset to evaluate on.
            loader (DataLoader): The DataLoader for the dataset.
            training_epoch (int): The current epoch of the training run
            is_aux_test_data (bool): Whether the dataset is an auxiliary test dataset.
            for_epochs (Optional[int]): The number of epochs to evaluate for. If None, evaluation is run once
            predictions_hook (Optional[Callable[[str, List[str], Tensor], None]]): A hook to call after each batch
            is_final_eval (bool): Whether this is the final evaluation of the model.

        Returns:
            metrics_tracker: The MetricsTracker instance containing the computed evaluation metrics.
        """

        self.model.eval()
        self.loss_tracker.reset()

        metrics_manager = self._get_metrics_manager(dataset, not is_aux_test_data)
        metrics_tracker = metrics_manager.add_tracker(self.split)
        dataset_prefix = dataset.get_name() if is_aux_test_data else None
        eval_epochs = for_epochs if for_epochs is not None else 1
        self.logger.info(f"Evaluating {dataset.get_name()}'s {self.split.name} set for {eval_epochs} epoch(s)...")
        amend_img_filenames = eval_epochs > 1
        for current_eval_epoch in range(1, eval_epochs + 1):
            with torch.no_grad():
                for ([images, masks], ids) in loader:
                    images = images.to(device=self.device, dtype=torch.float)
                    masks = masks.to(device=self.device, dtype=torch.long if dataset.is_multiclass() else torch.float)
                    img_filenames = [dataset.get_image_file_name(real_i) for real_i in ids]
                    if amend_img_filenames:
                        # if we are evaluating for multiple epochs, we need to amend the image filenames, otherwise
                        # metrics dicts for the respective img id will be overwritten every time
                        img_filenames = [
                            f"{img_filename.split('.')[0]}_epoch_{current_eval_epoch}.{img_filename.split('.')[1]}" for
                            img_filename in img_filenames]
                    predictions = self.compiled_model(images) if not hasattr(self.compiled_model,
                                                                             "medcam_dict") else self.compiled_model(
                        images, img_names=img_filenames, is_aux_test_data=is_aux_test_data,
                        xai_predictions_hook=xai_predictions_hook, dataset_name=dataset.get_name())
                    masks = masks.squeeze(1) if dataset.is_multiclass() else masks
                    loss = self.loss_func(predictions, masks)

                    self.loss_tracker.update(loss)
                    predictions = logits_to_segmentation_mask(
                        predictions).int() if dataset.is_multiclass() else predictions > 0.5

                    metrics_tracker.update_metrics_from_batch(img_filenames, predictions.cpu(), masks.cpu())
                    if predictions_hook is not None:
                        predictions_hook(dataset.get_name(), img_filenames, predictions.clone().cpu().int())

        metrics_tracker.compute_total_metrics()
        metrics_tracker.sync_to_tensorboard(training_epoch, ds_prefix=dataset_prefix)
        self.logger.info(f"Metrics for {self.split.name} set:")
        self.logger.info(metrics_tracker.get_metrics_summary())

        mean_loss = self.loss_tracker.compute_mean()
        self.loss_tracker.sync_to_tensorboard()
        metrics_manager.add_mean_loss(mean_loss, self.split, training_epoch)

        if self.save_sample_segmentations and eval_epochs <= 1 and is_final_eval:
            if for_epochs is not None and for_epochs > 1:
                self.logger.info(f"Evaluation is running in multi-epoch mode, so sample segmentations will only be "
                                 f"saved for the first epoch.")
            self.logger.info(f"Saving sample segmentations {dataset.get_name()}'s {self.split.name} set...")
            img_folder = f"images_{dataset.get_name().lower()}_{self.split.name}"
            img_save_pb = self.base_pb.clone().add(img_folder)
            img_size = self.cfg['architecture'].get('in_size', 512)
            visualizer = SegmentationVisualizer(self.model, self.device, dataset, img_save_pb, img_size)
            worst_ids = None
            if ImageSaveMode.WORST in self.save_modes:
                worst_ids = metrics_manager.get_bottom_k_img_ids(self.split, k=50)
            n_random_samples = 100 if ImageSaveMode.RANDOM_SUBSET in self.save_modes else None
            visualizer.save_segmentations(worst_ids, n_random_samples)
            del visualizer
        return metrics_manager

    @beartype
    def _get_metrics_manager(self, dataset: MedsegDataset, reuse_metrics_manager: bool = True) -> MetricsManager:
        if reuse_metrics_manager:
            return self.metrics_manager
        else:
            tbx_writer = self.metrics_manager.tbx_writer
            trial_name = self.metrics_manager.trial_name
            ds_prefix = dataset.get_name().lower()
            metrics_manager = MetricsManager(self.cfg,
                                             dataset.class_mapping,
                                             trial_name,
                                             tbx_writer,
                                             ds_prefix,
                                             base_pb=self.base_pb)
            metrics_manager.set_eval_object_sizes(self.eval_object_sizes, self.split)
            return metrics_manager


def get_model_info(cfg, dataset_manager, model, split) -> str:
    # Try to calculate FLOPs and MACs, otherwise just print the number of parameters
    try:
        flops, macs, params = calculate_flops(model=model,
                                              input_shape=(1, dataset_manager.get_dataset(split).img_channels,
                                                           cfg['architecture']['in_size'],
                                                           cfg['architecture']['in_size']),
                                              output_precision=4, print_results=False, print_detailed=False)
    except Exception as e:
        flops, macs, params = "N/A", "N/A", sum(p.numel() for p in model.parameters() if p.requires_grad)

    training_info = f"""
============================================
Evaluating model with the following settings:
Model:
    Architecture: {cfg['architecture']['arch_type']}
    Model name: {cfg['architecture']['model_name']}
    Learnable Parameters: {params}
    MACs: {macs}
    FLOPs: {flops}
"""
    return training_info
