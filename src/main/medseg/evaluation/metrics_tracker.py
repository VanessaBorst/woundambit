from collections import OrderedDict
from typing import List, Optional, Dict

import pandas as pd
import torch
from beartype import beartype
from tensorboardX import SummaryWriter
from torch import Tensor

from medseg.data.class_mapping import ClassMapping
from medseg.data.split_type import SplitType
from medseg.evaluation.metrics import EvalMetric, MetricsMethod, MetricsHistograms, compute_histograms, compute_metrics
from medseg.util.files import save_text_to_file


class MetricsTracker:
    """
    A class for tracking metrics during training and evaluation. It stores the metrics for each sample in an ordered
    dictionary.
    """

    @beartype
    def __init__(self,
                 tracked_metrics: List[EvalMetric],
                 split: SplitType,
                 trial_name: str,
                 class_mapping: ClassMapping,
                 ignore_background: bool = True,
                 method: MetricsMethod = MetricsMethod.MICRO,
                 tbx_writer: Optional[SummaryWriter] = None,
                 eval_object_sizes: bool = False
                 ) -> None:
        """
        Initialize the metrics tracker.

        Args:
            tracked_metrics (List[EvalMetric]): A list of evaluation metrics to track.
            class_mapping (ClassMapping): The class mapping.
            split (SplitType): The data split type.
            trial_name (str): The name of the trial.
            ignore_background (bool, optional): Whether to ignore the background class. Defaults to True.
            method (MetricsMethod, optional): The method for calculating metrics. Defaults to MetricsMethod.MICRO.
            tbx_writer (Optional[SummaryWriter], optional): A tensorboard summary writer. Defaults to None.
        """

        self.tracked_metrics = tracked_metrics
        self.split = split
        self.trial_name = trial_name
        self.class_mapping = class_mapping
        self.ignore_background = ignore_background
        self.method = method
        self.tbx_writer = tbx_writer
        # metrics: metric -> img_id -> value
        self.metrics: Dict[EvalMetric, OrderedDict[str: Tensor]] = {}
        for metric in self.tracked_metrics:
            self.metrics[metric] = OrderedDict()
        self.total_histograms = MetricsHistograms.empty(class_mapping.num_classes)
        self.total_metrics: Dict[EvalMetric, float] = {}
        self.histograms: OrderedDict[str, MetricsHistograms] = OrderedDict()
        self.eval_object_sizes = eval_object_sizes
        self.img_ids_mask_proportions: Dict[str, Dict[str, float]] = dict()  # img_id -> class name -> proportion

    @classmethod
    @beartype
    def from_dict(cls, metrics_dict: dict, tbx_writer: Optional[SummaryWriter] = None) -> 'MetricsTracker':
        """
        Restore metrics tracker from a dictionary.

        Args:
            metrics_dict (dict): A dictionary containing the metrics tracker state.
            tbx_writer (Optional[SummaryWriter], optional): the tensorboardX summary writer. Defaults to None.

        Returns:
            MetricsTracker: A restored MetricsTracker instance.
        """

        tracked_metrics = [EvalMetric(metric) for metric in metrics_dict['tracked_metrics']]
        metrics_tracker = cls(
            tracked_metrics,
            SplitType(metrics_dict['split']),
            metrics_dict['trial_name'],
            ClassMapping.from_dict(metrics_dict['class_mapping']),
            metrics_dict['ignore_background'],
            MetricsMethod(metrics_dict['method']),
            tbx_writer
        )
        if 'histograms' in metrics_dict:
            metrics_tracker.histograms = OrderedDict(
                {img_id: MetricsHistograms.from_state_dict(hist) for img_id, hist in
                 metrics_dict['histograms'].items()})
        if 'metrics' in metrics_dict:
            for metric in metrics_dict['metrics']:
                for img_id, value in metrics_dict['metrics'][metric].items():
                    metrics_tracker.metrics[EvalMetric(metric)][img_id] = value
        metrics_tracker.total_metrics = {EvalMetric(k): v for k, v in metrics_dict['total_metrics'].items()}
        metrics_tracker.eval_object_sizes = metrics_dict.get('eval_object_sizes',
                                                             metrics_tracker.eval_object_sizes)
        return metrics_tracker

    @beartype
    def state_dict(self, reduce: bool = True) -> dict:
        """
        Convert the metrics tracker to a dictionary.

        Args:
            reduce (bool, optional): Whether to reduce the state dictionary by not including the individual sample metrics and histograms. Defaults to True.

        Returns:
            dict: A dictionary containing the metrics tracker state.
        """

        state_dict = {
            'tracked_metrics': [metric.value for metric in self.tracked_metrics],
            'split': self.split.value,
            'trial_name': self.trial_name,
            'class_mapping': self.class_mapping.state_dict(),
            'ignore_background': self.ignore_background,
            'method': self.method.value,
            'total_histograms': self.total_histograms.state_dict(),
            'total_metrics': {metric.value: value for metric, value in self.total_metrics.items()},
            'eval_object_sizes': self.eval_object_sizes
        }
        if not reduce:
            state_dict['metrics'] = {metric.value: value for metric, value in self.metrics.items()}

            state_dict['histograms'] = {img_id: hist.state_dict() for img_id, hist in self.histograms.items()}

        return state_dict

    @beartype
    def to_df(self, reduce: bool = True) -> pd.DataFrame:
        total_metrics_str = {metric.value: value for metric, value in self.total_metrics.items()}

        data = {
            'total_histograms': [{k: v.tolist() for k, v in self.total_histograms.state_dict().items()}],
            'total_metrics': [total_metrics_str],
        }

        if not reduce:
            metrics_str = {metric.value: value for metric, value in self.metrics.items()}
            data['metrics'] = [metrics_str]
            data['histograms'] = [
                {key: {k: v.tolist() for k, v in hist.state_dict().items()} for key, hist in self.histograms.items()}]

        df = pd.DataFrame(data)

        # Storing metadata
        df.attrs['tracked_metrics'] = [metric.value for metric in self.tracked_metrics]
        df.attrs['split'] = self.split.value
        df.attrs['trial_name'] = self.trial_name
        df.attrs['class_defs'] = self.class_mapping.class_defs
        df.attrs['ignore_background'] = self.ignore_background
        df.attrs['method'] = self.method.value
        df.attrs['tbx_writer'] = self.tbx_writer

        # Add split and epoch columns
        df['split'] = self.split.value
        df['epoch'] = df.index

        return df

    @beartype
    def get_metrics_summary(self, metrics_to_print: Optional[List[EvalMetric]] = None) -> str:
        """
        Print the total metrics.

        Args:
            metrics_to_print (Optional[List[EvalMetric]], optional): A list of metrics to print. Defaults to None.
        """

        if metrics_to_print is None:
            metrics_to_print = self.tracked_metrics
        summary = " | ".join([f"{metric.value}: {self.total_metrics[metric]:.4f}" for metric in metrics_to_print if
                              metric in self.total_metrics])
        return summary


    @beartype
    def update_metrics_from_image(self, img_id: str, pred: Tensor, mask: Tensor) -> None:
        """
        Add a sample to the metrics tracker. Computes the per-sample metrics and stores them in the dictionary.

        Args:
            img_id (str): ID of the sample.
            pred (Tensor): Predicted segmentation mask.
            mask (Tensor): Ground truth segmentation mask.
            image (Tensor): Input image.
        """

        hists = compute_histograms(pred, mask, self.class_mapping.num_classes)
        # if img_id in self.histograms.keys():
        #     raise ValueError(f"Image with id {img_id} already in metrics tracker.")
        self.histograms[img_id] = hists
        self.total_histograms += hists
        sample_metrics: Dict[EvalMetric, Tensor] = compute_metrics(hists, self.tracked_metrics)
        for metric_key, metric_values in sample_metrics.items():
            self.metrics[metric_key][img_id] = metric_values

        if self.eval_object_sizes:
            self.eval_object_sizes_for_classes(mask, img_id)

    @beartype
    def eval_object_sizes_for_classes(self, mask: Tensor, img_id: str):
        if img_id not in self.img_ids_mask_proportions.keys():
            self.img_ids_mask_proportions[img_id] = dict()
            for class_def_dict in self.class_mapping.class_defs:
                class_label = class_def_dict['label']
                if class_label == 'background': continue
                class_pixel = class_def_dict['pixel_value']
                class_index = self.class_mapping.pixel_to_index(class_pixel)
                class_proportion = self.calculate_mask_proportion(mask, class_index)
                self.img_ids_mask_proportions[img_id][class_label] = class_proportion

    @beartype
    def calculate_mask_proportion(self, mask: Tensor, class_pixel: int) -> float:
        n_class_pixels = (mask == class_pixel).sum().item()
        n_total_pixels = mask.numel()
        return n_class_pixels / n_total_pixels

    @beartype
    def update_metrics_from_batch(self, img_filenames: List[str], preds: Tensor, masks: Tensor) -> None:
        """
        Add a batch of samples to the metrics tracker. Computes the per-sample metrics and stores them in the dictionary.

        Args:
            img_filenames (List[str]): IDs of the samples.
            preds (Tensor): Predicted segmentation masks.
            masks (Tensor): Ground truth segmentation masks.
        """

        for i, img_id in enumerate(img_filenames):
            self.update_metrics_from_image(img_id, preds[i], masks[i])

    @beartype
    def get_bottom_k_image_filenames(self, metric: EvalMetric, k: int = 10) -> List[str]:
        """
        Returns the image filenames of the k worst segmented images according to the given metric.

        Args:
            metric (EvalMetric): Metric to use for sorting.
            k (int, optional): Number of samples to return. Defaults to 1.

        Returns:
            List[str]: List image filenames with length k.
        """

        img_id_metrics = dict()
        ignore_class = self.class_mapping.bg_index if self.ignore_background else None
        if self.method == MetricsMethod.MICRO:
            k = min(k, len(self.histograms.keys()))
            for img_id, metrics_histograms in self.histograms.items():
                if self.method == MetricsMethod.MICRO:
                    hist = metrics_histograms.clone()
                    hist.sum_classes(ignore_class=ignore_class)
                    micro_avg_metric = compute_metrics(hist, [metric])[metric]
                    img_id_metrics[img_id] = micro_avg_metric.item()

        elif self.method == MetricsMethod.MACRO:
            k = min(k, len(self.metrics.keys()))
            for img_id in self.metrics[metric].keys():
                metrics_tensor = self.metrics[metric][img_id]
                if ignore_class is not None:
                    metrics_tensor = torch.cat((metrics_tensor[:ignore_class], metrics_tensor[ignore_class + 1:]))
                img_id_metrics[img_id] = metrics_tensor.mean().item()
        img_ids_ascending_by_metric = sorted(img_id_metrics, key=img_id_metrics.get)
        return img_ids_ascending_by_metric[:k]

    @beartype
    def get_bottom_k_image_filenames_for_class(self,
                                               metric: EvalMetric,
                                               k: int = 10,
                                               for_class: int = 1) -> List[str]:

        """
        Returns the filenames of the k images that have been segmented worst considering the given class and metric type.

        Args:
            metric (EvalMetric): Metric to use for sorting.
            k (int, optional): Number of samples to return. Defaults to 10.
            for_class (int, optional): The class for the metrics measurements. Defaults to 1.

        Returns:
            List[str]: List of k image filenames.
        """

        img_id_metrics = dict()

        k = min(k, len(self.metrics.keys()))
        for img_id in self.metrics[metric].keys():
            metrics_tensor = self.metrics[metric][img_id]
            if (len(metrics_tensor) - 1) < for_class:
                continue
            img_id_metrics[img_id] = metrics_tensor[for_class].item()

        img_ids_ascending_by_metric = sorted(img_id_metrics, key=img_id_metrics.get)
        return img_ids_ascending_by_metric[:k]

    @beartype
    def sync_to_tensorboard(self, epoch: int, tbx_writer: Optional[SummaryWriter] = None, ds_prefix=None) -> None:
        """
        Adds the metrics to the TensorBoard.

        Args:
            epoch (int): Current epoch.
            tbx_writer (Optional[SummaryWriter], optional): TensorBoard object. Defaults to None.
            ds_prefix ([type], optional): Dataset name prefix for use cases where metrics of multiple datasets are
                                         recorded. Defaults to None.
        """

        tbx_writer = tbx_writer or self.tbx_writer
        if tbx_writer is None:
            raise Warning("Could not sync metrics to TensorBoard as no SummaryWriter is set.")
        for metric_name, metric_value in self.total_metrics.items():
            metric_name = metric_name.value if isinstance(metric_name, EvalMetric) else metric_name
            if ds_prefix is None:
                tbx_writer.add_scalar(f"{self.split.value}/{metric_name}", metric_value, epoch)
            else:
                tbx_writer.add_scalar(f"{self.split.value}/{ds_prefix}/{metric_name}", metric_value, epoch)

    def get_per_sample_metric_list(self, metric: EvalMetric) -> List[float]:
        """
        Returns a list containing the metric for each sample.
        """
        metrics = []
        ignore_class = self.class_mapping.bg_index if self.ignore_background else None
        if self.method == MetricsMethod.MICRO:
            for img_id, metrics_histograms in self.histograms.items():
                hist = metrics_histograms.clone()
                hist.sum_classes(ignore_class=ignore_class)
                micro_avg_metric = compute_metrics(hist, [metric])[metric]
                metrics.append(micro_avg_metric.item())
        elif self.method == MetricsMethod.MACRO:
            for img_id in self.metrics[metric].keys():
                metrics_tensor = self.metrics[metric][img_id]
                if ignore_class is not None:
                    metrics_tensor = torch.cat((metrics_tensor[:ignore_class], metrics_tensor[ignore_class + 1:]))
                metrics.append(metrics_tensor.mean().item())
        return metrics

    @beartype
    def compute_total_metrics(self) -> None:
        """
        Computes the total metrics, i.e., the average of the per-sample metrics.
        """

        if self.method == MetricsMethod.MICRO:
            self.compute_total_metrics_micro_avg()
        elif self.method == MetricsMethod.MACRO:
            self.compute_total_metrics_macro_avg()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    @beartype
    def compute_total_metrics_micro_avg(self) -> dict:
        """
        Computes the total metrics, i.e., the average of the per-sample metrics using the micro-averaging method.

        Returns:
            dict: Dictionary containing the total metrics.
        """

        ignore_class = self.class_mapping.bg_index if self.ignore_background else None
        total_histograms = self.total_histograms.clone()
        total_histograms.sum_classes(ignore_class=ignore_class)
        total_micro_avg_metrics = compute_metrics(total_histograms, self.tracked_metrics)
        for metric_key, metric_class_values in total_micro_avg_metrics.items():
            total_micro_avg_metrics[metric_key] = metric_class_values.item()
        self.total_metrics = total_micro_avg_metrics
        return total_micro_avg_metrics

    @beartype
    def compute_total_metrics_macro_avg(self, set_total_metrics=True) -> dict:
        """
        Computes the total metrics, i.e., the average of the per-sample metrics using the macro-averaging method.

        Returns:
            dict: Dictionary containing the total metrics.
        """

        total_histograms = self.total_histograms.clone()
        if self.ignore_background:
            total_histograms.remove_class(self.class_mapping.bg_index)
        total_per_class_metrics = compute_metrics(total_histograms, self.tracked_metrics)
        total_macro_avg_metrics = {}
        for metric_key, metric_class_values in total_per_class_metrics.items():
            total_macro_avg_metrics[metric_key] = metric_class_values.mean().item()
        self.total_metrics = total_macro_avg_metrics
        return total_macro_avg_metrics

    @beartype
    def save_metrics_to_txt(self, path: str, current_epoch: int, header: Optional[str] = None) -> None:
        """
        Saves the current total metrics to a text file in an easily readable format. The text file includes the
        following information: trial_name, split, current epoch, metrics-method, the background ignore flag, and the
        total metrics.

        Args:
            path (str): The file path for the output text file.
            current_epoch (int): The current epoch during the training process.
            header (Optional[str], optional): Optional text to be added to the beginning of the file. Defaults to None.
        """

        content = []

        if header is not None:
            content.append(header)

        content.extend([
            f"Trial Name: {self.trial_name}",
            f"Split: {self.split.value}",
            f"Epoch: {current_epoch}",
            f"Metrics Method: {self.method.value}",
            f"Ignore Background: {self.ignore_background}",
            "",
            "Total Metrics:"
        ])

        for metric, value in self.total_metrics.items():
            content.append(f"{metric.value}: {value:.5f}")

        save_text_to_file("\n".join(content), path)
