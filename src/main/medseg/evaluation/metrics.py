from enum import Enum
from typing import Dict

import torch
from torch import Tensor


class EvalMetric(Enum):
    """
    Enum for the different evaluation metrics.
    """
    DICE = 'dice'
    IOU = 'iou'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'

    def get_abbr(self):
        if self.value == 'dice':
            return 'DSC'
        elif self.value == 'iou':
            return 'IoU'
        elif self.value == 'precision':
            return 'Prc'
        elif self.value == 'recall':
            return 'Rec'
        elif self.value == 'f1_score':
            return 'F1'
        else:
            raise ValueError('Unknown metric: {}'.format(self.value))

    def get_mean_abbr(self):
        return f"m{self.get_abbr()}"


class MetricsMethod(Enum):
    """
    Enum for the different methods to compute the evaluation metrics.
    """
    MICRO = 'micro'
    MACRO = 'macro'


class MetricsHistograms:
    """ Class for storing the histograms necessary to calculate the metrics. """

    def __init__(self, hist_intersect: Tensor, hist_union: Tensor, hist_pred: Tensor, hist_mask: Tensor):
        self.hist_intersect = hist_intersect
        self.hist_union = hist_union
        self.hist_pred = hist_pred
        self.hist_mask = hist_mask

    def __add__(self, other):
        return MetricsHistograms(self.hist_intersect + other.hist_intersect,
                                 self.hist_union + other.hist_union,
                                 self.hist_pred + other.hist_pred,
                                 self.hist_mask + other.hist_mask)

    @classmethod
    def empty(cls, num_classes):
        return cls(torch.zeros(num_classes), torch.zeros(num_classes), torch.zeros(num_classes),
                   torch.zeros(num_classes))

    @classmethod
    def from_state_dict(self, state_dict):
        return MetricsHistograms(state_dict['hist_intersect'], state_dict['hist_union'], state_dict['hist_pred'],
                                 state_dict['hist_mask'])

    def state_dict(self):
        return {'hist_intersect': self.hist_intersect, 'hist_union': self.hist_union, 'hist_pred': self.hist_pred,
                'hist_mask': self.hist_mask}

    def sum_classes(self, ignore_class=None):
        if ignore_class is not None:
            self.remove_class(ignore_class)
        self.hist_intersect = self.hist_intersect.sum(dim=0)
        self.hist_union = self.hist_union.sum(dim=0)
        self.hist_pred = self.hist_pred.sum(dim=0)
        self.hist_mask = self.hist_mask.sum(dim=0)

    def remove_class(self, class_id: int):
        self.hist_intersect = torch.cat((self.hist_intersect[:class_id], self.hist_intersect[class_id + 1:]))
        self.hist_union = torch.cat((self.hist_union[:class_id], self.hist_union[class_id + 1:]))
        self.hist_pred = torch.cat((self.hist_pred[:class_id], self.hist_pred[class_id + 1:]))
        self.hist_mask = torch.cat((self.hist_mask[:class_id], self.hist_mask[class_id + 1:]))

    def clone(self):
        return MetricsHistograms(self.hist_intersect.clone(), self.hist_union.clone(), self.hist_pred.clone(),
                                 self.hist_mask.clone())


def compute_metrics(histograms: MetricsHistograms, metrics: list) -> Dict[EvalMetric, Tensor]:
    """
    Compute metrics for either a single sample or the total metrics. The histograms can be either the total
    histograms or the histograms of a single sample. The histograms are expected to be 1D Tensors with each entry
    representing the number of pixels of a class in the predicted or ground truth segmentation mask. To calculate a
    single total metric across a dataset, the histograms should be summed up to a single value before calling this
    function.

    :param histograms: histograms of the sample
    :param metrics: list of metrics to compute
    :return: dictionary of the computed metrics
    """

    # loop over all metrics
    metrics_dict = {}
    hist_intersect = histograms.hist_intersect
    hist_union = histograms.hist_union
    hist_pred = histograms.hist_pred
    hist_mask = histograms.hist_mask

    for metric in metrics:
        if metric == EvalMetric.DICE:
            metrics_dict[EvalMetric.DICE] = compute_dice(hist_intersect, hist_pred, hist_mask)

        elif metric == EvalMetric.IOU:
            metrics_dict[EvalMetric.IOU] = compute_iou(hist_intersect, hist_union)

        elif metric == EvalMetric.PRECISION:
            metrics_dict[EvalMetric.PRECISION] = compute_precision(hist_intersect, hist_pred)

        elif metric == EvalMetric.RECALL:
            metrics_dict[EvalMetric.RECALL] = compute_recall(hist_intersect, hist_mask)

        elif metric == EvalMetric.F1_SCORE:
            precision = metrics_dict.get(EvalMetric.PRECISION, compute_precision(hist_intersect, hist_pred))
            recall = metrics_dict.get(EvalMetric.RECALL, compute_recall(hist_intersect, hist_mask))
            metrics_dict[EvalMetric.F1_SCORE] = compute_f1_score(precision, recall)
        else:
            raise ValueError(f'Unknown metric: {metric}')
    return metrics_dict


def compute_dice(hist_intersect: Tensor, hist_pred: Tensor, hist_mask: Tensor) -> Tensor:
    """
    Computes per-class Dice for a single sample. The computation method is mainly adapted from the mmsegmentation The
    computation method is mainly adapted from the mmsegmentation project (https://github.com/open-mmlab/mmsegmentation),
    which is used in many of the state-of-the-art image segmentation publications by Microsoft, NVIDIA, etc.

    :param hist_intersect: 1D tensor containing the number of correctly predicted pixels for each class
    :param hist_pred: 1D tensor containing the number of predicted pixels for each class
    :param hist_mask: 1D tensor containing the number of mask pixels for each class
    :return 1D tensor containing the Dice for each class
    """
    dice = 2 * hist_intersect / (hist_pred + hist_mask)
    dice[torch.isnan(dice)] = 0
    return dice


def compute_iou(hist_intersect: Tensor, hist_union: Tensor) -> Tensor:
    """
    Computes per-class IoU for a single sample. The computation method is mainly adapted from the mmsegmentation The
    computation method is mainly adapted from the mmsegmentation project (https://github.com/open-mmlab/mmsegmentation),
    which is used in many of the state-of-the-art image segmentation publications by Microsoft, NVIDIA, etc.

    :param hist_intersect: 1D tensor containing the number of correctly predicted pixels for each class
    :param hist_union: 1D tensor containing the union of the number of pixels for each class in the prediction and mask
    :return 1D tensor containing the IoU for each class
    """
    iou = hist_intersect / hist_union
    iou[torch.isnan(iou)] = 0
    return iou


def compute_precision(hist_intersect: Tensor, hist_pred: Tensor) -> Tensor:
    """
    Computes per-class precision for a single sample. The computation method is mainly adapted from the
    mmsegmentation project (
    https://github.com/open-mmlab/mmsegmentation), which is used in many of the state-of-the-art image segmentation
    publications by Microsoft, NVIDIA, etc.

    :param hist_intersect: 1D tensor containing the number of correctly predicted pixels for each class
    :param hist_pred: 1D tensor containing the number of pixels for each class in the prediction
    :return 1D tensor containing the precision for each class
    """
    precision = hist_intersect / hist_pred
    precision[torch.isnan(precision)] = 0
    return precision


def compute_recall(hist_intersect: Tensor, hist_mask: Tensor) -> Tensor:
    """
    Computes per-class recall for a single sample. The computation method is mainly adapted from the mmsegmentation
    project (https://github.com/open-mmlab/mmsegmentation), which is used in many of the state-of-the-art image
    segmentation publications by Microsoft, NVIDIA, etc.

    :param hist_intersect: 1D tensor containing the number of correctly predicted pixels for each class
    :param hist_mask: 1D tensor containing the number of pixels for each class in the mask
    :return: 1D tensor containing the recall for each class
    """
    recall = hist_intersect / hist_mask
    recall[torch.isnan(recall)] = 0
    return recall


def compute_f1_score(precision, recall) -> Tensor:
    """
    Computes per-class f1 score for a single sample. Note that, for binary segmentation tasks, the f1 score should
    yield the same result as the dice score. The computation method is mainly adapted from the mmsegmentation project
    (https://github.com/open-mmlab/mmsegmentation), which is used in many of the state-of-the-art image segmentation
    publications by Microsoft, NVIDIA, etc.

    :param precision: 1D tensor containing the precision for each class
    :param recall: 1D tensor containing the recall for each class
    :return: 1D tensor containing the f1 score for eachclass
    """
    f1_score = 2 * (precision * recall) / (precision + recall)
    # if precision and recall are both zero for any class, the tensor will have nan values, replace them with 0
    f1_score[torch.isnan(f1_score)] = 0
    return f1_score


def compute_histograms(pred: Tensor, mask: Tensor, num_classes) -> MetricsHistograms:
    """ Computes the intersection, union, prediction and mask histograms for each class. The histograms are 1D
    tensors with the i-th entry containing the number of pixels for class i. The computation method is mainly adapted
    from the mmsegmentation project (https://github.com/open-mmlab/mmsegmentation), which is used in many of the
    state-of-the-art image segmentation publications by Microsoft, NVIDIA, etc.

    :param pred: 2D tensor containing the predicted segmentation mask
    :param mask: 2D tensor containing the ground truth segmentation mask
    :param num_classes: number of classes in the segmentation task
    :return: Tuple of 1D tensors containing the intersection, union, prediction and mask histograms for each class
    """
    # 1D array containing an entry for each correctly predicted pixel. The entry is the class number
    intersect = pred[pred == mask]
    # counts the number of correctly predicted pixels for each class (including background)
    hist_intersect = torch.histc(intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
    # counts the number of pixels for each class in pred
    hist_pred = torch.histc(pred.float(), bins=num_classes, min=0, max=num_classes - 1)
    # counts the number of pixels for each class in mask
    hist_mask = torch.histc(mask.float(), bins=num_classes, min=0, max=num_classes - 1)
    # sums the number of pixels for each class in pred and mask, but substracts the number of correctly predicted pixels
    # in order to not double count them (since they are contained in pixels_pred_label and pixels_label)
    hist_union = hist_pred + hist_mask - hist_intersect
    return MetricsHistograms(hist_intersect, hist_union, hist_pred, hist_mask)
