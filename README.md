# WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care

Official implementation of the paper "WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care".

## Getting started
Dependencies can be installed with the following commands, using an anaconda or miniconda environment
    
```
conda create -n "woundambit" python=3.11
conda activate woundambit
pip3 install -e ./src/main
pip3 install -r requirements.txt
```

## Public Datasets
Datasets have to be downloaded manually and in case of the DFUC dataset require a registration.
However, converter scripts for the supported datasets can be found in `src/main/medseg/data/converters`.
The scripts are adapted to the folder structure of the downloaded dataset and should require no additional 
configuration. Please refer to the documentation in the respective converter script for more information.

Example:

```python3 ./src/main/medseg/data/converters/convert_cfu.py --in_path_dfuc="/home/user/Downloads/DFUC 2022 Data" --in_path_fuseg="/home/user/Downloads/FUSeg 2021 Data" --out_path="./data/datasets/cfu"```

The `out_path` argument should be set to the default path used in the respective dataset definition in `src/main/medseg/data/datasets` for the framework to find them without additional configuration.

## Own Datasets (OOD + Size Retrieval)
The out-of-distribution (OOD) dataset used in the publication to assess model generalizabilty can not be shared 
completely due to privacy regulations. 
However, with the requisite written consent, selected examples of the OOD dataset, along with all 20 images utilized for 
the purpose of evaluating the size retrieval performance, can be made available. 
These can be found in the directories labelled `./data/datasets/ood` and `./data/datasets/size_retrieval`, respectively.


## Pretrained ImageNet Weights and Trained Wound Segmentation Models

For information regarding the required pretrained ImageNet model weights, please refer to [this](docs/PretrainedWeights.md).
We provide the 5-fold cross-validation trained models that we used for the publication results on Zenodo under 
restricted access for double-blind review. Reviewers can access the model files via the following anonymous link: 
[Anonymous Zenodo Link](https://zenodo.org/records/15123641?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhiNTNlYTRhLTkyOWYtNDRhOC05OThlLTIxMGNiMzJkMTMzMiIsImRhdGEiOnt9LCJyYW5kb20iOiI3Njc1YjJiYzY5YTMzYmFmODRmYzhjYTViMTg0ZDI5MyJ9.Z0fZEKJUmA0QplsKMvaYOg6GhNGxOm_rfKT-H7GYDR4e6Arqd2d15FFuOunrqt79IPklYScBg26nUSqojpJA7A). The model weights will be made publicly available with a DOI upon publication.

## Detailed Model Configs
The configs that were used for creating the publication results can be found in `./configs` and
include details on the model architectures, training parameters, and more.

## Training
New training runs can be started with the following command:
`python3 ./src/main/medseg/training/train.py from_config --path="./configs/some_config.yaml"`
The training mode (hyperparameter optimization, k-fold) is determined by the config.

For logging and checkpoints saving, the `./out` folder is used, with subfolders according to the training type.
Within the subfolder corresponding to the training type, a new folder is created for each training run, according to the
`model_name` set in the training config and the timestamp at the start of the run. In this folder, the training log 
is saved along with checkpoints, a metric summary for each saved checkpoint, a model summary detailing the 
model architecture and parameter counts, the tensorboard event files for visualizing training and evaluation metrics, 
and more.

Interrupted hyperparameter optimization runs with grid search can be resumed with the following command:
`python3 ./src/main/medseg/training/train.py from_hyperopt_state --path="./some_folder/SegNeXtL-CFU-512-GridSearch.yaml" --grid_search_active `

For resuming a k-fold cross-validation run, the following command can be used:
`python3 ./src/main/medseg/training/train.py from_kfold_state --path="./some_folder/kfold_state.pkl"`

## Evaluation
Evaluations are automatically performed during and after training, however, separate evaluations can be performed with the following commands:

`python3 ./src/main/medseg/evaluation/eval.py from_checkpoint --path="./some_folder/example_checkpoint.pt" --split="test"`

`python3 ./src/main/medseg/evaluation/eval.py from_kfold --path="./some_folder/example_checkpoint.pt --add_aux_test_set="ood"`

U## Tools for Benchmarking, Size Retrieval, and Inference on Custom Images
Different tools that use the trained model checkpoints can be found in `./src/main/medseg/tools`.
In addition to the code for retrieving and evaluating wound sizes, there are tools for benchmarking 
different models and for performing inference with the trained AI models on custom images.
Please refer to the [dedicated tools documentation](docs/Tools.md) for more information.

## Final Notes

### Calflops with FUSegNet
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

### InternImage
For the usage of InternImage, DCNv3 needs to be installed manually.
Pre-compiled `.whl` files can be found [here](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files).
Alternatively, it can be compiled as described in the installation instructions of the 
[official repository](https://github.com/OpenGVLab/InternImage/tree/master/segmentation). 

For our setting (Python 3.11.11., Pytorch 2.0.1, CUDA 11.7 ), we upload the pre-compiled `.whl` file that we created 
for the DCNv3 operation in the `./data/wheel` folder.

### TransNeXt
The [official GitHub repository](https://github.com/DaiShiResearch/TransNeXt/) offers a CUDA implementation for 
TransNeXt, which can be installed as described in the corresponding ReadMe.

For our setup (Python 3.11.11, Pytorch 2.0.1, CUDA 11.7 ), we upload the pre-compiled `.whl` file that we created 
for the SWAttention operation in the `./data/wheel` folder.
