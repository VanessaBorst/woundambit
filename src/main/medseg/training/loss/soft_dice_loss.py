import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    """
    SoftDiceLoss used in FCBFormer, from: https://github.com/ESandML/FCBFormer
    """

    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2
        score = (
                2.0
                * (intersection.sum(1) + self.smooth)
                / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score


class MultiClassSoftDiceLoss(nn.Module):
    # experimental, needs to be tested / evaluated
    def __init__(self, smooth: float = 1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        targets = targets.unsqueeze(1)
        targets_one_hot = torch.zeros_like(logits).scatter_(1, targets, 1)

        probs = torch.sigmoid(logits)
        dice_losses = []
        for cls in range(num_classes):
            m1 = probs[:, cls, :, :].contiguous().view(-1)
            m2 = targets_one_hot[:, cls, :, :].contiguous().view(-1)
            intersection = (m1 * m2).sum()
            dice_loss = 1 - (2. * intersection + self.smooth) / (m1.sum() + m2.sum() + self.smooth)
            dice_losses.append(dice_loss)

        return torch.stack(dice_losses).mean()
