import numpy as np
from tensorboardX import SummaryWriter
from torch import Tensor

from medseg.data.split_type import SplitType


class LossTracker:
    def __init__(self, split: SplitType, tbx_writer: SummaryWriter = None):
        self.split = split
        self.losses = []
        self.mean_losses_per_epoch = []
        self.tbx_writer = tbx_writer

    def update(self, loss: Tensor):
        self.losses.append(loss.item())

    def compute_mean(self) -> float:
        mean_loss = np.mean(self.losses)
        self.mean_losses_per_epoch.append(mean_loss)
        return mean_loss

    def get_last(self):
        return self.losses[-1]

    def sync_to_tensorboard(self, epoch: int = None, tbx_writer=None):
        epoch = len(self.mean_losses_per_epoch) if epoch is None else epoch
        metric_value = self.mean_losses_per_epoch[epoch - 1]
        tbx_writer = self.tbx_writer if tbx_writer is None else tbx_writer
        if tbx_writer is not None:
            tbx_writer.add_scalar(f"{self.split.value}/mean_loss", metric_value, epoch)
        else:
            print("Could not sync metrics from LossTracker to tensorboard as no SummaryWriter is set.")

    def reset(self):
        self.losses = []

    def state_dict(self):
        return {
            "losses": self.losses,
            "mean_losses_per_epoch": self.mean_losses_per_epoch
        }

    def load_state_dict(self, state_dict):
        self.losses = state_dict["losses"]
        self.mean_losses_per_epoch = state_dict["mean_losses_per_epoch"]
