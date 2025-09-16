# --------------------------------------------------------
# Code adapted by Vanessa Borst, Timo Dittus and Contributors
# from https://github.com/YuWenLo/HarDNet-DFUS
#
# Original license from the HarDNet-DFUS repository: Not specified
# --------------------------------------------------------


import torch
from torch import nn
from torch.nn import functional as F


# refactored original loss code from HarDNet repository and split it up into two loss functions because it had
# lots of unused code and was really confusing to read (like everything else in that codebase)
class HarDNetDFUSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        weights_tensor = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
        weighted_bce = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        weighted_bce = (weights_tensor * weighted_bce).sum(dim=(2, 3)) / weights_tensor.sum(dim=(2, 3))
        pred = torch.sigmoid(pred)
        inter = ((pred * gt) * weights_tensor).sum(dim=(2, 3))
        union = (pred * weights_tensor).sum(dim=(2, 3)) + (gt * weights_tensor).sum(dim=(2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)
        loss = weighted_bce.mean() + iou.mean()
        return loss


class HarDNetDFUSBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.half).requires_grad_(
            False).cuda() if torch.cuda.is_available() else \
            torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.half).requires_grad_(False)

    def forward(self, pred, gt):
        pred = nn.Sigmoid()(pred)
        # Cast laplacian to the same dtype as gt to avoid an error in case the model is NOT trained with mixed precision
        # (RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same)
        gt = (F.conv2d(gt.float(), self.laplacian.to(gt.dtype), stride=1, padding=1) > 0.1).float()
        bce_loss = F.binary_cross_entropy_with_logits(pred, gt.expand(gt.size(0), pred.size(1), gt.size(2), gt.size(3)))
        return bce_loss
