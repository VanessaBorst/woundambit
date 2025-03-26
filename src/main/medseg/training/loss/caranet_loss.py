import torch
from torch import nn

from torch.nn import functional as F
# CaraNet loss, adapted with modifications from https://github.com/AngeLouCN/CaraNet
# very similar to HarDNet loss, seemingly they both copied vast amounts of code from the PraNet publication
class CaraNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
        wbce = F.binary_cross_entropy_with_logits(pred, gt, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * gt) * weit).sum(dim=(2, 3))
        union = ((pred + gt) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()
