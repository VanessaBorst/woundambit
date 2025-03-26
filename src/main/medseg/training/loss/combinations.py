from torch import nn, Tensor

from medseg.training.loss.soft_dice_loss import SoftDiceLoss


class BCEDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()

    def forward(self, predictions: Tensor, targets: Tensor):
        return self.bcel(predictions, targets) + self.dice(predictions, targets)


# class BDiceFocal(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.focal = BinaryFocalLoss()
#         self.dice = SoftDiceLoss()
#
#     def forward(self, predictions: Tensor, targets: Tensor):
#         return self.focal(predictions, targets) + self.dice(predictions, targets)
#
#
# class BCEFocal(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bcel = nn.BCEWithLogitsLoss()
#         self.focal = BinaryFocalLoss()
#
#     def forward(self, predictions: Tensor, targets: Tensor):
#         return self.bcel(predictions, targets) + self.focal(predictions, targets)
#
#
# class BCEFocalDice(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bcel = nn.BCEWithLogitsLoss()
#         self.dice = SoftDiceLoss()
#         self.focal = BinaryFocalLoss()
#
#     def forward(self, predictions: Tensor, targets: Tensor):
#         return self.bcel(predictions, targets) + self.dice(predictions, targets) + self.focal(predictions, targets)
