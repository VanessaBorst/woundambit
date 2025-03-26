import torch
import torch.nn as nn
from torch import Tensor


# Focal loss implementation from
class BinaryFocalTverskyLogitsLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def tversky_index(self, y_true, y_pred):
        y_true_pos = y_true.view(y_true.size(0), -1)  # flatten spatial and channel dimensions
        y_pred_pos = y_pred.view(y_pred.size(0), -1)
        true_pos = torch.sum(y_true_pos * y_pred_pos, dim=1)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos), dim=1)
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos, dim=1)
        return (true_pos + self.smooth) / (
                    true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        pt_1 = self.tversky_index(y_true, y_pred)
        return torch.mean(torch.pow((1 - pt_1), self.gamma))

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return self.focal_tversky(torch.sigmoid(x), target)
