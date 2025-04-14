# WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care

This repository contains the official implementation of the paper:

**"WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care"**,  
accepted to the Applied Data Science Track at **ECML PKDD 2025**.

If you find this repository helpful for your work, please cite our contribution as follows:

```bibtex
@inproceedings{borst2025woundambit,
  title = {WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care},
  author = {Borst, Vanessa and Dittus, Timo and Dege, Tassilo and Schmieder, Astrid and Kounev, Samuel},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track},
  publisher = {Springer Nature Switzerland},
  year = {2025}
}
```


## 🔧 Getting Started

Dependencies can be installed with the following commands, using an anaconda or miniconda environment
    
```
conda create -n "woundambit" python=3.11
conda activate woundambit
pip3 install -e ./src/main
pip3 install -r requirements.txt
```

If a GPU is available, set the `CUDA_VISIBLE_DEVICES` environment variable to the GPU ID before running the code:
```bash
export CUDA_VISIBLE_DEVICES=<GPU_ID>
```

For the usage of FUSegNet, TransNeXt, and InternImage, please look at the 
[model-specific installation remarks](docs/ModelSpecifics.md).


## 🎯 Pretrained ImageNet Weights 

The framework uses pretrained ImageNet weights for all models, which need to be downloaded manually 
in advance for most architectures.
For more details regarding the required ImageNet weights, please refer to the [respective documentation](docs/PretrainedWeights.md).

> For the architectures used in our publication, pretrained weights that are not downloaded automatically are already included under `data/pretrained`.


## 🧠  Trained Models (5-Fold CV)
The trained 5-fold wound segmentation models are available on [Zenodo](https://zenodo.org/records/15123641).

To use the below tools for inference on custom images, benchmarking or size retrieval, please download the trained models,
unpack them, and place the `k_fold_models` folder in the `./out` folder.

## ⚙️ Tools: Benchmarking, Size Retrieval, and Inference on Custom Images
Different tools that use the trained model checkpoints can be found in `./src/main/medseg/tools`.
In addition to the code for retrieving and evaluating wound sizes, there are tools for benchmarking 
different models and for performing inference with the trained AI models on custom images.
Please refer to the [dedicated tools documentation](docs/Tools.md) for more information.

## 📁 Datasets

### Public Data
Datasets have to be downloaded manually and in case of the DFUC dataset require a registration.
However, converter scripts for the supported datasets can be found in `src/main/medseg/data/converters`.
The scripts are adapted to the folder structure of the downloaded dataset and should require no additional 
configuration. Please refer to the documentation in the respective converter script for more information.

Example:

```python3 ./src/main/medseg/data/converters/convert_cfu.py --in_path_dfuc="/home/user/Downloads/DFUC 2022 Data" --in_path_fuseg="/home/user/Downloads/FUSeg 2021 Data" --out_path="./data/datasets/cfu"```

The `out_path` argument should be set to the default path used in the respective dataset definition in `src/main/medseg/data/datasets` for the framework to find them without additional configuration.

### Own Data (OOD + Size Retrieval)
The out-of-distribution (OOD) dataset used in the publication to assess model generalizabilty can not be shared 
completely due to privacy regulations. 
However, with the requisite written consent, selected examples of the OOD dataset, along with all 20 images utilized for 
the purpose of evaluating the size retrieval performance, can be made available. 
These can be found in the directories labelled `./data/datasets/ood` and `./data/datasets/size_retrieval`, respectively.

📝 : Within the code, the OOD dataset is referred to as `UkwTest`.


## 🧪 Configs and Custom Training
The configs that were used for creating the publication results can be found in `./configs` and
include details on the model architectures, training parameters, and more. To make them work, the specified datasets 
must exist (see above section for how to create them).

For details regarding the training of new models and the subsequent evaluation,
please refer to the [custom training and evaluation documentation](docs/CustomTrainingAndEval.md).


## 📦 Supplementary Material and Resources

Additional files such as supplementary analysis, archived code snapshot, and trained model weights are hosted on **Zenodo**:

- 📄 [Supplementary PDF](https://zenodo.org/records/15673941)
- 🧠 [Trained Models](https://zenodo.org/records/15123641)
- 🗂️ [Archived Code Snapshot]()

All files are provided under an appropriate open-source or research license.
