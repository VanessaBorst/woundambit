# Model-Specific Installation Remarks

## Calflops with FUSegNet
The framework uses the calflops library for calculating the number of FLOPs and GMACs in the models.
In order to prevent a TypeError that is caused by using the library together with FuSegNet and its encoder EfficientNet,
the following block has to be added to `calflops/pytorch_ops.py` in the `_conv_flops_compute` function before line 94:
```
### Manual addition - if format is [1,1] instead of (1,1), then we need to convert to tuple
stride = tuple(stride) if type(stride) is list and len(stride) == 2 else stride
### End manual addition
```
Explanation: The EfficientNet encoder in FuSegNet, or more precisely the depth-wise convolution operation within the MBConv block, uses a list for the stride that is not supported by the library (e.g., [1, 1] instead of (1,1)). 
The added block converts the list into a tuple to prevent the TypeError.

## InternImage
For the usage of InternImage, DCNv3 needs to be installed manually.
Pre-compiled `.whl` files can be found [here](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files).
Alternatively, it can be compiled as described in the installation instructions of the 
[official repository](https://github.com/OpenGVLab/InternImage/tree/master/segmentation). 

For our setting (Python 3.11.11., Pytorch 2.0.1, CUDA 11.7 ), we upload the pre-compiled `.whl` file that we created 
for the DCNv3 operation in the `./data/wheel` folder.

## TransNeXt
The [official GitHub repository](https://github.com/DaiShiResearch/TransNeXt/) offers a CUDA implementation for 
TransNeXt, which can be installed as described in the corresponding ReadMe.

For our setup (Python 3.11.11, Pytorch 2.0.1, CUDA 11.7 ), we upload the pre-compiled `.whl` file that we created 
for the SWAttention operation in the `./data/wheel` folder.