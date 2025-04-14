import gc
import os
from copy import deepcopy
from logging import Logger
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from beartype import beartype
from scipy.stats import stats
from torch import Tensor

from medseg.data.datasets.medseg_dataset import MedsegDataset
from medseg.data.split_type import SplitType
from medseg.evaluation.evaluator import Evaluator
from medseg.evaluation.metrics import EvalMetric, MetricsMethod
from medseg.evaluation.metrics_manager import MetricsManager
from medseg.evaluation.metrics_tracker import MetricsTracker
from medseg.util.date_time import get_current_date_time_str
from medseg.util.files import save_text_to_file
from medseg.util.logger import create_custom_logger
from medseg.util.path_builder import PathBuilder
from medseg.util.random import ensure_reproducibility
from medseg.evaluation.medcam.medcam_utils import save_attention_map


class EnsembleEvaluator:
    @beartype
    def __init__(self, checkpoint_paths: List[str],
                 save_pb: PathBuilder,
                 eval_split_main=SplitType.TEST,
                 logger: Optional[Logger] = None,
                 additional_aux_datasets: Optional[List[str]] = None,
                 use_xai: bool = False):
        # for path in checkpoint_paths:
        #     assert os.path.exists(path), f"Checkpoint path {path} does not exist"
        self.checkpoint_paths = checkpoint_paths
        self.additional_aux_datasets = additional_aux_datasets
        self.use_xai = use_xai
        self.cfgs = self._fetch_cfgs()
        self._check_and_prepare_cfgs()
        self.eval_split_main = eval_split_main
        self.datasets = dict()
        self.save_pb = save_pb
        self.logger = logger if logger is not None else create_custom_logger("Ensemble evaluation",
                                                                             save_pb.clone().build())
        # if all eval splits are the same, we can do a majority vote on the main dataset
        indices_key = f"{eval_split_main.value.lower()}_indices"
        main_dataset_indices = [cfg['dataset'].get(indices_key, None) for cfg in self.cfgs]
        self.do_majority_vote_on_main_dataset = all([main_dataset_indices[0] == main_dataset_indices[i] \
                                                     for i in range(1, len(main_dataset_indices))])

        # predictions are stored in a dict with the dataset name as key and the value being a list with one item per
        # checkpoint, where each item is a dict with the filename as key and the prediction as value.
        self.xai_predictions: Dict[str, List[Dict[str, np.ndarray]]] = dict()
        self.predictions: Dict[str, List[Dict[str, np.ndarray]]] = dict()
        self.metrics_managers: Dict[str, List[MetricsManager]] = dict()
        self.metrics_trackers_ens_avg: Dict[str, MetricsTracker] = dict()
        self.metrics_trackers_ens_mv: Dict[str, MetricsTracker] = dict()
        self.i = 0

    @classmethod
    @beartype
    def from_kfold_state(cls, state_dict: dict, path: str, additional_aux_datasets: Optional[List[str]] = None, use_xai: bool = False):
        cfg = state_dict['cfg']
        trained_model_paths = cfg["k_fold"]["trained_model_paths"]
        # for i, trained_model_path in enumerate(trained_model_paths):
        #     parent_folder = os.path.basename(os.path.dirname(trained_model_path))
        #     trained_model_paths[i] = os.path.join(path, parent_folder, os.path.basename(trained_model_path))
        for trained_model_path in trained_model_paths:
            assert os.path.exists(trained_model_path), f"Checkpoint path {trained_model_path} does not exist"

        main_eval_split = SplitType.TEST if cfg['k_fold'].get('include_test_split', False) else SplitType.VAL
        save_pb = PathBuilder.from_path(path)

        log_filename = f"ensemble_evaluation_log_{get_current_date_time_str()}.txt"
        logger = create_custom_logger('ensemble_evaluation_logger', save_pb.clone().add(log_filename).build())

        return EnsembleEvaluator(trained_model_paths, save_pb, main_eval_split, logger,
                                 additional_aux_datasets, use_xai)

    @beartype
    def _fetch_cfgs(self) -> List[dict]:
        # load all checkpoints and only fetch the cfg
        cfgs = list()
        for path in self.checkpoint_paths:
            checkpoint_dict = torch.load(path, map_location=torch.device('cpu'))
            cfgs.append(deepcopy(checkpoint_dict['cfg']))
            del checkpoint_dict
            self.free_memory()
        return cfgs

    @beartype
    def run_evaluation(self) -> Tuple[Dict[str, MetricsTracker], Dict[str, MetricsTracker]]:
        for i, path in enumerate(self.checkpoint_paths):
            self.i = i
            self.logger.info(f"Evaluating Checkpoint {i + 1}/{len(self.checkpoint_paths)}...")
            # on the auxiliary test datasets, a majority vote eval can be additionally performed
            # on other models, the metrics from test datasets are averaged
            # print(os.environ['CUDA_VISIBLE_DEVICES'])
            checkpoint_dict = torch.load(path, map_location=torch.device("cuda"))
            checkpoint_dict['cfg'] = self.cfgs[i]
            evaluator = Evaluator.from_checkpoint(checkpoint_dict, path=path, split=self.eval_split_main, use_XAI=self.use_xai)
            evaluator.logger = self.logger
            # main dataset
            ensure_reproducibility(self.cfgs[i]['settings']['random_seed'])
            ds_main, loader_main = evaluator.dataset_manager.get_dataset_and_loader(self.eval_split_main)
            self.datasets[ds_main.get_name()] = ds_main
            pred_hook_main_dataset = self.predictions_hook if self.do_majority_vote_on_main_dataset else None
            metrics_manager_main = evaluator.evaluate_dataset(ds_main, loader_main, 0, False, 1, pred_hook_main_dataset)

            # create list if no there yet
            if ds_main.get_name() not in self.metrics_managers:
                self.metrics_managers[ds_main.get_name()] = list()
            self.metrics_managers[ds_main.get_name()].append(metrics_manager_main)
            self.metrics_trackers_ens_avg[ds_main.get_name()] = self._get_tracker_from_mm(metrics_manager_main,
                                                                                          self.eval_split_main)
            # aux datasets
            if evaluator.dataset_manager.has_aux_test_datasets():
                evaluator.split = SplitType.TEST  # for aux datasets, whole set is always in test split
                ensure_reproducibility(self.cfgs[i]['settings']['random_seed'])
                for ds_aux, loader_aux in evaluator.dataset_manager.get_aux_test_datasets_and_loaders():
                    self.datasets[ds_aux.get_name()] = ds_aux
                    metrics_manager_aux = evaluator.evaluate_dataset(ds_aux, loader_aux, 0, True, 1,
                                                                     self.predictions_hook, xai_predictions_hook=self.xai_predictions_hook)
                    if ds_aux.get_name() not in self.metrics_managers:
                        self.metrics_managers[ds_aux.get_name()] = list()
                    self.metrics_managers[ds_aux.get_name()].append(metrics_manager_aux)
                    self.metrics_trackers_ens_avg[ds_aux.get_name()] = self._get_tracker_from_mm(metrics_manager_aux,
                                                                                                 SplitType.TEST)

            del evaluator
            del checkpoint_dict
            self.free_memory()
        self.save_main_eval_results()
        if self.use_xai:
            self.run_majority_vote_xai()
        self.run_majority_vote_evaluation()
        self.logger.info("Ensemble evaluation finished.")
        return self.metrics_trackers_ens_avg, self.metrics_trackers_ens_mv

    @beartype
    def save_main_eval_results(self):
        for ds_name, mm_list in self.metrics_managers.items():
            split = self.datasets[ds_name].split_type
            metrics = dict()
            txt = [
                f"Ensemble Evaluation Results",
                "---------------------------------------------",
                f"Dataset: {ds_name}",
                f"Split: {split.value}",
            ]
            for i, metrics_manager in enumerate(mm_list):
                metrics_manager.base_pb = self.save_pb.clone()
                tracker = metrics_manager.get_last_tracker(split)
                if i == 0:
                    txt.extend([
                        f"Metrics Method: {tracker.method.value}",
                        f"Ignore Background: {tracker.ignore_background}",
                        "---------------------------------------------",
                        "Metrics for each checkpoint:"
                    ])

                txt.extend(["---------------------------------------------",
                            f"Checkpoint {i + 1}:"])
                for metric, val in tracker.total_metrics.items():
                    txt.append(f"{metric.value}: {val:.5f}")
                    metrics[metric.value] = val if metric.value not in metrics else metrics[metric.value] + val
            txt.append("---------------------------------------------")
            txt.append("Average Metrics over all checkpoints:")
            for metric, val in metrics.items():
                txt.append(f"{metric}: {val / len(mm_list):.5f}")
            txt.append("---------------------------------------------")
            txt_save_path = self.save_pb.clone().add(f"ensemble_eval_{ds_name}_{split.value}.txt").build()
            save_text_to_file("\n".join(txt), txt_save_path)

    def run_majority_vote_evaluation(self):
        for dataset_name, models_preds in self.predictions.items():
            self.logger.info(f"Starting majority vote evaluation for {dataset_name}...")
            n_models = len(models_preds)
            assert n_models > 1, "Ensemble evaluation: Majority vote evaluation requires at least two models."
            assert all([models_preds[0].keys() == models_preds[i].keys() for i in range(1, n_models)]), \
                f"Ensemble evaluation: Mask predictions for {dataset_name} must have the same prediction filenames."
            dataset = self.datasets[dataset_name]
            metrics_tracker = self._get_tracker_from_ds(dataset, self.cfgs[0])
            img_keys = list(models_preds[0].keys())
            for img_key in img_keys:
                mask_preds = [models_preds[i][img_key] for i in range(n_models)]
                mask_preds_majority = self.mask_majority_vote(mask_preds)
                real_i = dataset.all_images.index(img_key)
                ds_i = dataset.real_index_to_dataset_index(real_i)
                _, mask, _ = dataset.__getitem__(ds_i)
                pred = torch.from_numpy(mask_preds_majority)
                while len(mask.shape) > len(pred.shape):
                    pred = pred.unsqueeze(0)
                pred = pred.type(mask.type())
                metrics_tracker.update_metrics_from_image(img_key, pred, mask)
            metrics_tracker.compute_total_metrics()
            txt_path = self.save_pb.clone().add(
                f"ensemble_eval_{dataset_name}_{dataset.split_type.value}_majority_vote_metrics.txt").build()
            metrics_tracker.save_metrics_to_txt(txt_path, 0)
            self.metrics_trackers_ens_mv[dataset_name] = metrics_tracker
            self.logger.info(f"Majority vote metrics for {dataset_name} set:")
            self.logger.info(metrics_tracker.get_metrics_summary())
        self.logger.info(f"Majority vote evaluation finished! Results saved to {self.save_pb.clone().build()}")

    def run_majority_vote_xai(self):
        for dataset_name, layers in self.xai_predictions.items():
            for layer, models_preds in layers.items():
                self.logger.info(f"Starting majority vote for xai evaluation for {dataset_name}...")
                n_models = len(models_preds)
                assert n_models > 1, "Ensemble evaluation: Majority vote evaluation requires at least two models."
                assert all([models_preds[0].keys() == models_preds[i].keys() for i in range(1, n_models)]), \
                    f"Ensemble evaluation: XAI predictions for {dataset_name} must have the same prediction filenames."
                dataset = self.datasets[dataset_name]
                img_keys = list(models_preds[0].keys())
                for img_key in img_keys:
                    xai_preds = [models_preds[i][img_key] for i in range(n_models)]
                    xai_preds_majority = self.xai_mean_vote(xai_preds)
                    real_i = dataset.all_images.index(img_key)
                    ds_i = dataset.real_index_to_dataset_index(real_i)
                    image, mask, _ = dataset.__getitem__(ds_i)
                    pred = torch.from_numpy(xai_preds_majority)
                    save_path = self.save_pb.clone().add(
                        f"aux_xai_eval_{dataset_name}_{dataset.split_type.value}_{layer}_mean_vote").add(img_key).build()
                    save_attention_map(filename=save_path, attention_map=pred, heatmap=True, raw_input=image)





    @beartype
    def _get_tracker_from_mm(self, mm: MetricsManager, split: SplitType) -> Optional[MetricsTracker]:
        tracker = mm.get_last_tracker(split)
        if tracker is None:
            print(f"WARNING: No metrics tracker found for {split.value}")
        return tracker

    @beartype
    def _get_tracker_from_ds(self, dataset: MedsegDataset, cfg: dict) -> MetricsTracker:
        tracked_metrics = cfg['metrics']['tracked']
        tracked_metrics = [EvalMetric(metric) if isinstance(metric, str) else metric for metric in tracked_metrics]
        split = dataset.split_type
        trial_name = f"k_fold_ensemble_evaluation_{dataset.get_name().lower()}"
        ignore_background = cfg['metrics'].get('ignore_background_class', True)
        metrics_method = cfg['metrics'].get('averaging_method', MetricsMethod.MICRO)
        if isinstance(metrics_method, str):
            metrics_method = MetricsMethod(metrics_method.lower())
        metrics_tracker = MetricsTracker(tracked_metrics, split, trial_name, dataset.class_mapping, ignore_background,
                                         metrics_method, eval_object_sizes=True)
        return metrics_tracker

    @staticmethod
    @beartype
    def mask_majority_vote(mask_preds: List[np.ndarray]) -> np.ndarray:
        mask_preds_int = [arr.astype(int) for arr in mask_preds]
        majority_mask, _ = stats.mode(mask_preds_int, axis=0, keepdims=True)
        majority_mask = np.squeeze(majority_mask)
        return majority_mask

    @staticmethod
    @beartype
    def xai_mean_vote(mask_preds: list[np.ndarray]) -> np.ndarray:
        mask_preds = np.array(mask_preds)
        mean_mask = np.mean(mask_preds, axis=0)
        return np.squeeze(mean_mask)


    @beartype
    def predictions_hook(self, dataset_name: str, img_filenames: List[str], predictions: Tensor):
        assert len(img_filenames) == 1, "Ensemble evaluation: Only one image per batch is supported."
        if dataset_name not in self.predictions:
            self.predictions[dataset_name] = [dict() for _ in range(len(self.checkpoint_paths))]
        num_classes = self.datasets[dataset_name].class_mapping.num_classes
        if num_classes <= 2:
            predictions = predictions.numpy().astype('bool')
        elif num_classes <= 256:
            predictions = predictions.numpy().astype('uint8')
        else:
            predictions = predictions.numpy().astype('uint16')

        self.predictions[dataset_name][self.i][img_filenames[0]] = predictions

    @beartype
    def xai_predictions_hook(self, dataset_name: str, img_filenames: List[str], predictions: dict):
        assert len(img_filenames) == 1, "Ensemble evaluation: Only one image per batch is supported."
        if dataset_name not in self.xai_predictions:
            self.xai_predictions[dataset_name] = {
                layer: [dict() for _ in range(len(self.checkpoint_paths))]
                for layer in predictions.keys()
            }
        for layer, prediction in predictions.items():
            self.xai_predictions[dataset_name][layer][self.i][img_filenames[0]] = prediction



    @beartype
    def free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    @beartype
    def _check_and_prepare_cfgs(self):
        def print_warning(value_name: str, value: any):
            print(f"Ensemble evaluation: Difference in '{value_name}' of the given checkpoints detected.")
            print(f"Using {value_name} {str(value)} of the first checkpoint for all checkpoints.")

        test_transforms = [cfg['transforms']['test'] for cfg in self.cfgs]
        # check equality of all transforms
        if not all([test_transforms[0] == test_transforms[i] for i in range(1, len(test_transforms))]):
            print_warning("test transform pipeline", '')
            for i in range(1, len(test_transforms)): self.cfgs[i]['transforms']['test'] = test_transforms[0]

        random_seeds = [cfg['settings']['random_seed'] for cfg in self.cfgs]
        if not all([random_seeds[0] == random_seeds[i] for i in range(1, len(random_seeds))]):
            print_warning("random seed", random_seeds[0])
            for i in range(1, len(random_seeds)):
                self.cfgs[i]['settings']['random_seed'] = random_seeds[0]

        dataset_types = [cfg['dataset']['type'] for cfg in self.cfgs]
        if not all([dataset_types[0] == dataset_types[i] for i in range(1, len(dataset_types))]):
            print("Ensemble evaluation: Difference in dataset types of the given checkpoints detected.")
            print(f"Using the dataset type {dataset_types[0]} of the first checkpoint for all checkpoints.")
            print_warning("dataset type", dataset_types[0])
            for i in range(1, len(dataset_types)):
                self.cfgs[i]['dataset']['type'] = dataset_types[0]

        aux_datasets = [cfg['dataset'].get('aux_test_datasets', None) for cfg in self.cfgs]
        if not all([aux_datasets[0] == aux_datasets[i] for i in range(1, len(aux_datasets))]):
            print_warning("auxiliary test datasets", aux_datasets[0])
            for i in range(1, len(aux_datasets)):
                self.cfgs[i]['dataset']['aux_test_datasets'] = aux_datasets[0]

        metrics = [cfg['settings'].get('metrics', None) for cfg in self.cfgs]
        if not all([metrics[0] == metrics[i] for i in range(1, len(metrics))]):
            print_warning("metrics", metrics[0])
            for i in range(1, len(metrics)):
                self.cfgs[i]['settings']['metrics'] = metrics[0]

        # set default settings
        for cfg in self.cfgs:
            cfg['settings']['eval_object_sizes'] = True
            cfg['settings']['num_workers'] = 1
            cfg['settings']['batch_size'] = 1
            cfg['settings']['final_eval_epochs'] = 1
            if self.additional_aux_datasets is not None and len(self.additional_aux_datasets) > 0:
                aux_datasets = cfg['dataset'].get('aux_test_datasets', list())
                aux_datasets.extend(self.additional_aux_datasets)
                cfg['dataset']['aux_test_datasets'] = aux_datasets

