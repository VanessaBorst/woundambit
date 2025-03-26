import torch

from medseg.evaluation.params import create_model_summary
from medseg.models.segmentors.fusegnet.fusegnet_helpers import Unet
from medseg.models.segmentors.segmentor import Segmentor

# Adapted from https://github.com/mrinal054/FUSegNet
# Note: The authors use a modified version of segmentation_models_pytorch, which is not available in the official
# repository. Therefore, we need to copy some of the adapted code in fusegnet_helpers.py and fusegnet_decoder.py


BASE_MODEL = 'FuSegNet'  # give any name for the model
ENCODER = 'efficientnet-b7'  # encoder model
ENCODER_WEIGHTS = 'imagenet'  # encoder weights


ACTIVATION = 'sigmoid'  # output activation. sigmoid for binary and softmax for multi-class segmentation

TO_CATEGORICAL = False  # if True, converts to onehot
RAW_PREDICTION = False  # if true, then stores raw predictions (i.e. before applying threshold)


class FUSegNet(Segmentor):
    def __init__(self,
                 in_size=512,
                 in_channels=3,
                 out_channels=1):
        super(FUSegNet, self).__init__()
        self.in_size = in_size

        self.model = Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=in_channels,
            classes=out_channels,
            # TODO: Dynamically adapt this based on loss function
            activation=None,    # Use None for logits (since the loss during our experiments is BCEWithLogitsLoss)
            decoder_attention_type='pscse',
        )

    def forward(self, x):
        x = self.model(x)
        # return torch.sigmoid(x)
        return x

    #TODO: Add default loss of FuSegNet (Sum of Dice and Focal Loss)


if __name__ == "__main__":
    im = torch.randn(1, 3, 512, 512)
    model = FUSegNet(in_channels=3, out_channels=1)
    print(create_model_summary(model, im.shape,depth=6))
