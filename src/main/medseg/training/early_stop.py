from typing import List

from medseg.evaluation.metrics import EvalMetric
from medseg.evaluation.metrics_tracker import MetricsTracker


class EarlyStop:
    """A class implementing an early stopping mechanism for model training.
    Since the early stopping mechanism is based on one of the tracked metrics,
    the optimization direction is assumed to be maximization.

    Attributes:
        metric (EvalMetric): The metric used to monitor model performance.
        tolerance (int): The number of epochs with no improvement before stopping.
        min_delta (float): The minimum change in the monitored metric to qualify as an improvement.
        counter (int): A counter for the number of epochs with no improvement.
        stop_triggered (bool): Whether early stopping has been triggered.
    """

    def __init__(self, metric=EvalMetric.IOU, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.metric = metric if isinstance(metric, EvalMetric) else EvalMetric(metric)
        self.best_checkpoint_metric_value = 0
        self.counter = 0
        self.stop_triggered = False

    def check_metric(self, metrics_trackers: List[MetricsTracker]):
        """Checks if the metric has improved and updates the counter.

        Args:
            metrics_trackers (List[MetricsTracker]): A list of metrics trackers for previous epochs.
        """
        if metrics_trackers is None or len(metrics_trackers) < 2:
            return
        if self.metric not in metrics_trackers[-1].tracked_metrics \
                or self.metric not in metrics_trackers[-2].tracked_metrics:
            raise Warning(f"Early stop metric {self.metric} is not tracked in the given metrics tracker")

        metric_current = metrics_trackers[-1].total_metrics[self.metric]
        # metric_previous = metrics_trackers[-2].total_metrics[self.metric]

        # Before: metric_current - metric_previous
        # print(f"Current metric: {metric_current}, Best metric: {self.best_checkpoint_metric_value}")
        # print(f"Delta: {metric_current - self.best_checkpoint_metric_value}")

        # Check if there has been an improvement in the metric compared to the best checkpoint that is
        # at least min_delta
        if (metric_current - self.best_checkpoint_metric_value) >= self.min_delta:
            # print("Resetting counter")
            self.counter = 0
        else:
            self.counter += 1
            print(f"Counter increased: {self.counter}")
            if self.counter >= self.tolerance:
                # print("Early stopping triggered")
                self.stop_triggered = True

        # Finally, update the best checkpoint metric value
        self.best_checkpoint_metric_value = max(self.best_checkpoint_metric_value, metric_current)

    # TODO: In future release, all variables should be saved in a dictionary
    def state_dict(self):
        """Returns the state dictionary of the EarlyStop instance.

        Returns:
            dict: A dictionary containing the state of the EarlyStop instance.
        """
        return {
            'counter': self.counter,
            'stop_triggered': self.stop_triggered,
            'best_checkpoint_metric_value': self.best_checkpoint_metric_value
        }

    def load_state_dict(self, state_dict: dict):
        """Initializes an EarlyStop instance from a state dictionary.

        Args:
            state_dict (dict): The state dictionary to initialize the EarlyStop instance.
        """
        self.counter = state_dict['counter']
        self.stop_triggered = state_dict['stop_triggered']
        self.best_checkpoint_metric_value = state_dict['best_checkpoint_metric_value']


def get_early_stop(cfg: dict):
    """Returns an EarlyStop instance based on the provided configuration.

    Args:
        cfg (dict): The configuration dictionary containing early stopping settings.

    Returns:
        EarlyStop: An EarlyStop instance or None if early stopping is not enabled.
    """
    if cfg['early_stop'] is None or cfg['early_stop'] is False:
        return None
    elif cfg['early_stop'] is True:
        return EarlyStop()
    else:
        return EarlyStop(**cfg['early_stop'])
