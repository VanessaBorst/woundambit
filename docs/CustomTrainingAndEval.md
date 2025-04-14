# Custom Training and Evaluation

## Preparations
For the example configs to work, the **underlying datasets must exist**.
Alternatively, custom datasets can be created and specified in the config files.

## Training
New training runs can be started with the following command:
```bash
python3 ./src/main/medseg/training/train.py from_config --path="./configs/some_config.yaml"
```
The training mode (hyperparameter optimization, k-fold) is determined by the config.

For logging and checkpoints saving, the `./out` folder is used, with subfolders according to the training type.
Within the subfolder corresponding to the training type, a new folder is created for each training run, according to the
`model_name` set in the training config and the timestamp at the start of the run. In this folder, the training log 
is saved along with checkpoints, a metric summary for each saved checkpoint, a model summary detailing the 
model architecture and parameter counts, the tensorboard event files for visualizing training and evaluation metrics, 
and more.

Interrupted hyperparameter optimization runs with grid search can be resumed with the following command:
```bash
python3 ./src/main/medseg/training/train.py from_hyperopt_state --path="./some_folder/SegNeXtL-CFU-512-GridSearch.yaml" --grid_search_active
```

For resuming a k-fold cross-validation run, the following command can be used:
```bash
python3 ./src/main/medseg/training/train.py from_kfold_state --path="./some_folder/kfold_state.pkl"
```

## Evaluation
Evaluations are automatically performed during and after training, however, separate evaluations can be performed with the following commands:
```bash

python3 ./src/main/medseg/evaluation/eval.py from_checkpoint --path="./some_folder/example_checkpoint.pt" --split="test"

python3 ./src/main/medseg/evaluation/eval.py from_kfold --path="./some_folder/example_checkpoint.pt --add_aux_test_set="UkwTest"
```
