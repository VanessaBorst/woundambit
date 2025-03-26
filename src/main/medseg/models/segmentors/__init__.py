# Important that this is here, otherwise the class mapping in the builder won't function!

# If new architectures are added, the main neural net module of the architecture should be imported here
# the framework will match the name from the yaml config file (key architecture -> type) to the class name
# given here (e.g. "FCBFormer").
from .fcbformer.fcbformer import FCBFormer
from .hardnet.hardnet_dfus import HarDNetDFUS
from .fusegnet.fusegnet import FUSegNet
from .hiformer.hiformer import HiFormer, HiFormerS, HiFormerB, HiFormerL
from .segformer.segformer import SegformerB0, SegformerB1, SegformerB2, SegformerB3, SegformerB4, SegformerB5, Segformer
from .segnext.segnext import SegNeXtT, SegNeXtS, SegNeXtT, SegNeXtB, SegNeXtL, SegNeXt
from .unet.unet import UNet
from .internimage_upernet.internimage_upernet import InternImageUperNet, InternImageUperNet_T, InternImageUperNet_S, \
    InternImageUperNet_B, InternImageUperNet_L, InternImageUperNet_XL, InternImageUperNet_H
from .missformer.missformer import MISSFormer
from .vwformer.vwformer_MiT import VWFormerMiTB0, VWFormerMiTB1, VWFormerMiTB2, VWFormerMiTB3,\
    VWFormerMiTB4, VWFormerMiTB5, VWFormerMiT
from .vwformer.vwformer_convnext import VWFormerConvNextS, VWFormerConvNextB, VWFormerConvNext
from .transnext_upernet.transnext_upernet import TransNeXtUperNet_Base, TransNeXtUperNet_Small, TransNeXtUperNet_Tiny