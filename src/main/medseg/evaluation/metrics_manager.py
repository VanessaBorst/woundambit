# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import pickle
from typing import Dict, OrderedDict
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from beartype import beartype
from tensorboardX import SummaryWriter
from torch import Tensor

from medseg.data.class_mapping import ClassMapping
from medseg.data.split_type import SplitType
from medseg.evaluation.metrics import EvalMetric, MetricsMethod
from medseg.evaluation.metrics_tracker import MetricsTracker
from medseg.evaluation.plots import plot_metrics
from medseg.util.files import save_text_to_file
from medseg.util.path_builder import PathBuilder


class MetricsManager:
    @beartype
    def __init__(self, cfg: dict,
                 class_mapping: ClassMapping,
                 trial_name: str,
                 tbx_writer: Optional[SummaryWriter] = None,
                 ds_prefix: Optional[str] = None,
                 base_pb: Optional[PathBuilder] = None,
                 eval_object_sizes: bool = False
                 ):

        self.cfg = cfg
        self.base_pb = base_pb if base_pb is not None else PathBuilder.trial_out_builder(cfg)
        tracked_metrics_cfg_available = not (cfg['metrics'] is None or cfg['metrics']['tracked'] is None or len(
            cfg['metrics']['tracked']) == 0)
        self.tracked_metrics = cfg['metrics']['tracked'] if tracked_metrics_cfg_available else [m for m in EvalMetric]
        self.metrics_method = cfg['metrics'][
            'averaging_method'] if tracked_metrics_cfg_available else MetricsMethod.MICRO

        self.train_metrics_trackers: List[MetricsTracker] = []
        self.val_metrics_trackers: List[MetricsTracker] = []
        self.test_metrics_trackers: List[MetricsTracker] = []

        self.metrics_trackers: Dict[SplitType, List[MetricsTracker]] = {
            SplitType.TRAIN: self.train_metrics_trackers,
            SplitType.VAL: self.val_metrics_trackers,
            SplitType.TEST: self.test_metrics_trackers
        }

        self.mean_loss: Dict[SplitType, Dict[int, float]] = {
            SplitType.TRAIN: {},
            SplitType.VAL: {},
            SplitType.TEST: {}
        }

        self.class_mapping = class_mapping
        self.trial_name = trial_name
        self.ignore_background_class = 'metrics' in cfg and cfg['metrics'].get('ignore_background_class', True)
        self.tbx_writer = tbx_writer
        self.ds_prefix = ds_prefix
        self.eval_object_sizes = eval_object_sizes
        self.eval_object_sizes_split = SplitType.TEST

    @beartype
    def set_eval_object_sizes(self, eval_object_sizes: bool, split: SplitType):
        self.eval_object_sizes = eval_object_sizes
        self.eval_object_sizes_split = split
        for tracker in self.metrics_trackers[split]:
            tracker.eval_object_sizes = eval_object_sizes

    @beartype
    def state_dict(self, reduce: bool = True):
        return {
            'tracked_metrics': [m.value for m in self.tracked_metrics],
            'metrics_method': self.metrics_method.value,
            'metrics_trackers': {
                SplitType.TRAIN.value: [tracker.state_dict(reduce=reduce) for tracker in self.train_metrics_trackers],
                SplitType.VAL.value: [tracker.state_dict(reduce=reduce) for tracker in self.val_metrics_trackers],
                SplitType.TEST.value: [tracker.state_dict(reduce=reduce) for tracker in self.test_metrics_trackers]
            },
            'mean_loss': {
                SplitType.TRAIN.value: {e: l for e, l in self.mean_loss[SplitType.TRAIN].items()},
                SplitType.VAL.value: {e: l for e, l in self.mean_loss[SplitType.VAL].items()},
                SplitType.TEST.value: {e: l for e, l in self.mean_loss[SplitType.TEST].items()},
            },
            'trial_name': self.trial_name,
            'class_mapping': self.class_mapping.state_dict(),
            'ignore_background_class': self.ignore_background_class,
            'ds_prefix': self.ds_prefix,
            'eval_object_sizes': self.eval_object_sizes
        }

    @beartype
    def to_df(self, reduce: bool = True) -> pd.DataFrame:
        metrics_dfs = []
        for split, trackers in self.metrics_trackers.items():
            for epoch, tracker in enumerate(trackers):
                # No need to call tracker.to_df(reduce=reduce)
                for metric, value in tracker.total_metrics.items():
                    metrics_dfs.append({
                        'split': split.value,
                        'epoch': epoch,
                        'metric': metric.value,
                        'value': value
                    })

        metrics_df = pd.DataFrame(metrics_dfs)

        mean_losses = []
        for split, losses in self.mean_loss.items():
            for epoch, loss in losses.items():
                mean_losses.append({
                    'split': split.value,
                    'epoch': epoch,
                    #'mean_loss': loss
                    'metric': 'mean_loss',
                    'value': loss
                })

        mean_loss_df = pd.DataFrame(mean_losses)
        # Merge the metrics and mean loss data
        # result_df = pd.merge(metrics_df, mean_loss_df, on=['split', 'epoch'], how='outer')
        result_df = pd.concat([metrics_df, mean_loss_df], ignore_index=True)

        return result_df

    @beartype
    def load_state_dict(self, state_dict: dict):
        self.tracked_metrics = [EvalMetric(m) for m in state_dict['tracked_metrics']]
        self.metrics_method = MetricsMethod(state_dict['metrics_method'])

        self.train_metrics_trackers = [MetricsTracker.from_dict(tracker_dict, tbx_writer=self.tbx_writer)
                                       for tracker_dict in state_dict['metrics_trackers'][SplitType.TRAIN.value]]
        self.val_metrics_trackers = [MetricsTracker.from_dict(tracker_dict, tbx_writer=self.tbx_writer)
                                     for tracker_dict in state_dict['metrics_trackers'][SplitType.VAL.value]]
        self.test_metrics_trackers = [MetricsTracker.from_dict(tracker_dict, tbx_writer=self.tbx_writer)
                                      for tracker_dict in state_dict['metrics_trackers'][SplitType.TEST.value]]

        self.metrics_trackers = {
            SplitType.TRAIN: self.train_metrics_trackers,
            SplitType.VAL: self.val_metrics_trackers,
            SplitType.TEST: self.test_metrics_trackers
        }
        self.mean_loss = {
            SplitType.TRAIN: {e: l for e, l in state_dict['mean_loss'][SplitType.TRAIN.value].items()},
            SplitType.VAL: {e: l for e, l in state_dict['mean_loss'][SplitType.VAL.value].items()},
            SplitType.TEST: {e: l for e, l in state_dict['mean_loss'][SplitType.TEST.value].items()},
        }
        self.trial_name = state_dict['trial_name']
        self.ignore_background_class = state_dict['ignore_background_class']
        self.ds_prefix = state_dict.get('ds_prefix', self.ds_prefix)
        self.eval_object_sizes = state_dict.get('eval_object_sizes', self.eval_object_sizes)
        # disable for now as the class mapping is set in __init__ anyway (and might be the better fitting version)
        # self.class_mapping = ClassMapping.from_dict(state_dict['class_mapping'])

    @beartype
    def get_last_metric(self, split: SplitType, metric: EvalMetric) -> float:
        return self.get_last_tracker(split).total_metrics[metric]

    @beartype
    def get_last_tracker(self, split: SplitType) -> MetricsTracker:
        return self.metrics_trackers[split][-1]

    @beartype
    def add_tracker(self, split: SplitType):
        eval_object_sizes = self.eval_object_sizes if split == SplitType.TEST else False
        new_tracker = MetricsTracker(tracked_metrics=self.tracked_metrics,
                                     split=split,
                                     trial_name=self.trial_name,
                                     class_mapping=self.class_mapping,
                                     method=self.metrics_method,
                                     tbx_writer=self.tbx_writer,
                                     eval_object_sizes=eval_object_sizes)
        self.metrics_trackers[split].append(new_tracker)
        return new_tracker

    @beartype
    def add_mean_loss(self, loss_value: float, split: SplitType, epoch: int):
        self.mean_loss[split][epoch] = loss_value

    @beartype
    def get_last_mean_loss(self, split: SplitType) -> Optional[float]:
        epochs = self.mean_loss[split].keys()
        if len(epochs) == 0:
            return None
        max_epoch = max(epochs)
        return self.mean_loss[split][max_epoch]

    @beartype
    def get_bottom_k_img_ids(self, split: SplitType, metric: EvalMetric = EvalMetric.DICE, k: int = 10) -> \
            List[str]:
        fallback_metrics = [EvalMetric.DICE, EvalMetric.IOU, EvalMetric.F1_SCORE, EvalMetric.PRECISION,
                            EvalMetric.RECALL]
        i = 0
        while metric not in self.tracked_metrics:
            metric = fallback_metrics[i]
            i += 1
        return self.get_last_tracker(split).get_bottom_k_image_filenames(metric, k)

    @beartype
    def create_eval_object_sizes_latex_table(self,
                                             img_id_metrics_dict: Dict[EvalMetric, OrderedDict[str, Tensor]],
                                             img_ids_mask_proportions: Dict[str, Dict[str, float]],
                                             class_label: str,
                                             metrics_to_include: List[EvalMetric],
                                             n_bins=5):
        img_ids = []
        mask_proportions = {}

        for img_id, class_dict in img_ids_mask_proportions.items():
            if class_label in class_dict.keys():
                img_ids.append(img_id)
                mask_proportions[img_id] = class_dict[class_label]

        # Define bins
        bins = np.linspace(min(mask_proportions.values()), max(mask_proportions.values()), n_bins + 1)
        labels = [f"{round(bins[i], 2) * 100}\%-{round(bins[i + 1], 2) * 100}\%" for i in
                  range(n_bins)]  # ranges for the x-axis

        class_index = self.class_mapping.class_label_to_index(class_label)
        metric_bins = {metric.value: [] for metric in metrics_to_include}
        for img_id in img_ids:
            mask_prop_bin = pd.cut([mask_proportions[img_id]], bins, labels=labels)[0]
            for metric in metrics_to_include:
                if pd.isna(mask_prop_bin): continue
                metric_bins[metric.value].append((mask_prop_bin, img_id_metrics_dict[metric][
                    img_id][class_index].item()))

        # Construct DataFrame
        data = {metric.value: [0] * n_bins for metric in metrics_to_include}
        counts = {metric.value: [0] * n_bins for metric in metrics_to_include}
        df = pd.DataFrame(data, index=labels)
        for metric, values in metric_bins.items():
            for bin_label, value in values:
                df.loc[bin_label, metric] += value
                counts[metric][labels.index(bin_label)] += 1

        # Compute means
        for metric in metrics_to_include:
            df[metric.get_mean_abbr()] = [f"{(val / count):.4f}" if count != 0 else 0 for val, count in
                                          zip(df[metric.value], counts[metric.value])]

        # Generate LaTeX table
        latex_table = df.style.to_latex(hrules=True)
        return latex_table

    @beartype
    def save_full_metrics(self, reduce: bool = True, to_csv: bool = True):
        file_base_name = 'metrics_manager_dump'
        if self.ds_prefix is not None:
            file_base_name += f'_{self.ds_prefix}'

        if to_csv:
            path = self.base_pb.clone().add(f"{file_base_name}.csv").build()
            self.to_df(reduce=reduce).to_csv(path, index=False)

        else:
            path = self.base_pb.clone().add(f"{file_base_name}.pkl").build()
            state = self.state_dict(reduce)
            with open(path, 'wb') as file:
                pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)

    @beartype
    def save_plots(self, split: SplitType, metrics_to_plot: List[Union[EvalMetric, str]] = None):
        if metrics_to_plot is None:
            metrics_to_plot = self.tracked_metrics
        else:
            metrics_to_plot = [m for m in metrics_to_plot if m in self.tracked_metrics or m == 'mean_loss']
        path = self.base_pb.clone().add('plots').add(self.ds_prefix).add(split.value).build()
        plot_metrics(self.to_df(), split, metrics_to_plot, path, single_plot=False)

    @beartype
    def sync_hparams_to_tensorboard(self, hparams: dict, split: SplitType, cur_epoch: int):

        metrics_tb = {}
        metrics_for_tb = self.get_last_tracker(split).total_metrics

        for metric_key, metric_value in metrics_for_tb.items():
            metric_name = metric_key.value if isinstance(metric_key, EvalMetric) else metric_key
            metrics_tb[f"{split.value}/final_{metric_name}"] = metric_value

        metrics_tb[f"{split.value}/final_mean_loss"] = self.get_last_mean_loss(SplitType(split.value))
        metrics_tb["train/total_epochs"] = cur_epoch

        tb_path = self.base_pb.clone().tb().build()
        # convert unsupported types to string
        hparams = {k: str(v) if isinstance(v, (list, tuple, dict)) else v for k, v in hparams.items()}
        self.tbx_writer.add_hparams(hparams, metrics_tb, name=tb_path)

    @beartype
    def save_metrics_txt(self, split: SplitType, prefix: str, cur_epoch: int):
        last_metrics_tracker = self.get_last_tracker(split)
        if last_metrics_tracker is None:
            return
        txt_header = f"--- {split.get_full_name} metrics summary ---"
        path = self.base_pb.clone().add(f"{prefix}_{split.value.lower()}_metrics.txt").build()
        last_metrics_tracker.save_metrics_to_txt(path, cur_epoch, txt_header)

        if self.eval_object_sizes and self.eval_object_sizes_split == split:
            img_ids_mask_proportions = last_metrics_tracker.img_ids_mask_proportions
            if img_ids_mask_proportions is None or len(img_ids_mask_proportions.keys()) == 0:
                return
            img_id_metrics_dict = last_metrics_tracker.metrics
            tracked_metrics = last_metrics_tracker.tracked_metrics
            for class_def in self.class_mapping.class_defs:
                class_label = class_def['label']
                if class_label == 'background': continue
                latex_table = self.create_eval_object_sizes_latex_table(img_id_metrics_dict,
                                                                        img_ids_mask_proportions,
                                                                        class_label,
                                                                        tracked_metrics)
                path = self.base_pb.clone().add('metrics_mask_proportions').add(
                    f"{prefix}_{split.value.lower()}_{class_label}.txt").build()
                save_text_to_file(latex_table, path)
