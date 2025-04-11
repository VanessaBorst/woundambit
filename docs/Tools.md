# WoundAmbit: Tools Overview

## Benchmarking
This tool can be used to evaluate key computational metrics of the different AI models, including:
- **GMACs**: The number of Giga Multiply-Accumulate operations required to process an image of size 512x512 (determined with `calflops`).
- **Model Size**: The size of the model in terms of trainable parameters (in M, determined with `calflops`).
- **Inference Time**: The average time (in ms) taken by the model to process an image and generate a prediction.
- **Throughput**: The number of images processed by the model during inference (in images per second).
- **Peak GPU Memory**: The maximum GPU memory (in MB) used by the model during inference.

### Input
The script expects the trained model checkpoints from the 5-fold CV to be located in the `out/k_fold_models` directory.
For this, download the model checkpoints from [Zenodo](https://zenodo.org/records/15123641?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhiNTNlYTRhLTkyOWYtNDRhOC05OThlLTIxMGNiMzJkMTMzMiIsImRhdGEiOnt9LCJyYW5kb20iOiI3Njc1YjJiYzY5YTMzYmFmODRmYzhjYTViMTg0ZDI5MyJ9.Z0fZEKJUmA0QplsKMvaYOg6GhNGxOm_rfKT-H7GYDR4e6Arqd2d15FFuOunrqt79IPklYScBg26nUSqojpJA7A), 
maintain the internal folder structure, and place everything in the required directory after unpacking.

By default, two different datasets are used for benchmarking, averaging results afterwards:
- **ukw**: The ood dataset used in the publication, which is not (fully) publicly available.
- **cfu**: The custom CFU dataset that is a combination of the DFUC 2022 and FUSeg 2021 datasets.

If you want to use your own dataset, you can specify the path at the beginning of the `main` method.
Moreover, you can change the default number of images (300), the number of warm-up iterations (10), and the seed (42).

### Output
Among others, the script generates different TEX files containing the GMACs, model size, average inference time, 
average throughput, and the peak GPU memory usage for each model. 
It creates a dedicated file per dataset and another file containing the average values across these datasets, using the
specified `out_path` (default: `./out/benchmarking`).

## Size Retrieval 

This tool can be used to analyze pairs of 1) images containing our reference object with ArUco markers and 2) AI-predicted wound masks to 
accurately determine wound dimensions. 
For more details, see the [Size Retrieval Tool](SizeRetrieval_and_Eval.md) document.

## Inference on Custom Images

This tool can be used to do inference with the trained AI models on a folder of custom images.
By default, the script uses the 5-fold CV trained models from the `out/k_fold_models` directory, which can 
be downloaded and unpacked from [Zenodo](https://zenodo.org/records/15123641?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhiNTNlYTRhLTkyOWYtNDRhOC05OThlLTIxMGNiMzJkMTMzMiIsImRhdGEiOnt9LCJyYW5kb20iOiI3Njc1YjJiYzY5YTMzYmFmODRmYzhjYTViMTg0ZDI5MyJ9.Z0fZEKJUmA0QplsKMvaYOg6GhNGxOm_rfKT-H7GYDR4e6Arqd2d15FFuOunrqt79IPklYScBg26nUSqojpJA7A).