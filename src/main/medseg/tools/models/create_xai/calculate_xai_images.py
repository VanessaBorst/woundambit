import os
import pickle
from typing import Dict, List

import click
import numpy as np
import torch
import torchvision

from medseg.data.split_type import SplitType
from medseg.tools.models.create_xai.xai_utils import (
    _load_kfold_model_from_state,
    _check_and_prepare_cfgs,
    _load_model_from_checkpoint,
    prepare_normal_cfg,
    create_path_builder,
    create_xai_images,
    run_majority_vote_xai
)

# Disable torchvision beta transforms warning
torchvision.disable_beta_transforms_warning()


@click.group()
def create_xai():
    pass


@create_xai.command(name='from_checkpoint')
# Paths to single checkpoint files
@click.option('--paths', type=str, required=True, multiple=True)
@click.option("--num_img", type=int, required=True, default=20)
@click.option("--split", type=str, required=False, default="test")
def create_xai_from_checkpoint(paths: str, num_img: int, split: str = "test"):
    assert paths is not None, "No paths given"
    paths = list(paths)
    print("Start benchmarking from checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = SplitType(split.lower())
    for i, path in enumerate(paths):
        checkpoint = torch.load(path)
        cfg = prepare_normal_cfg(checkpoint['cfg'])
        base_pb = create_path_builder().add(cfg['architecture']['model_name'])
        model, dataset_manager, dataset = _load_model_from_checkpoint(checkpoint=checkpoint, device=device)
        real_num_img = min(num_img, len(dataset_manager.get_dataset_and_loader(split)[1]))
        create_xai_images(model, dataset_manager, split, base_pb, real_num_img)


@create_xai.command(name='from_kfold')
# Paths to the kfold state files
@click.option('--paths', type=click.Path(exists=True, dir_okay=False), multiple=True, required=True)
@click.option("--num_img", type=int, required=True, default=0)
@click.option('--add_aux_test_set', type=str, required=False, multiple=True)
@click.option("--aux_num_img", type=int, required=False, default=350)
def create_xai_from_kfold(paths, num_img, add_aux_test_set, aux_num_img):
    assert paths is not None, "No paths given"
    paths = list(paths)
    if add_aux_test_set is not None:
        add_aux_test_set = list(add_aux_test_set)
    for i, path in enumerate(paths):
        state = pickle.load(open(path, 'rb'))
        path = os.path.dirname(path)
        assert os.path.isabs(path), f"Path must be absolute, but is: {path}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_paths, split = _load_kfold_model_from_state(state)
        cfgs, model_name = _check_and_prepare_cfgs(model_paths, add_aux_test_set)
        base_pb = create_path_builder().add(model_name)
        xai_predictions: Dict[str, List[Dict[str, np.ndarray]]] = dict()
        datasets = dict()

        for fold, model_path in enumerate(model_paths, start=1):
            def xai_predictions_hook(dataset_name: str, img_filenames: List[str], predictions: dict):
                assert len(img_filenames) == 1, "Ensemble evaluation: Only one image per batch is supported."
                if dataset_name not in xai_predictions:
                    xai_predictions[dataset_name] = {
                        layer: [dict() for _ in range(len(model_paths))]
                        for layer in predictions.keys()
                    }
                for layer, prediction in predictions.items():
                    xai_predictions[dataset_name][layer][fold - 1][img_filenames[0]] = prediction

            model, dataset_manager, dataset_name = _load_model_from_checkpoint(model_path, device, cfgs[fold-1])
            fold_pb = base_pb.clone().add(dataset_name).add(f"fold_{fold}")
            real_num_img = min(num_img, len(dataset_manager.get_dataset_and_loader(split)[1]))
            dataset = create_xai_images(model, dataset_manager, split, fold_pb, real_num_img)
            datasets[dataset_name] = dataset

            if dataset_manager.aux_test_datasets_and_loaders is not None:
                for i, aux_dataset_manager in enumerate(dataset_manager.aux_test_datasets_and_loaders):
                    aux_dataset = cfgs[fold - 1]['dataset']['aux_test_datasets'][i]
                    aux_real_num_img = min(aux_num_img, len(aux_dataset_manager[0].all_images))
                    dataset = create_xai_images(model, aux_dataset_manager, SplitType.TEST, fold_pb, aux_real_num_img,
                                                xai_predictions_hook, aux_dataset, True)
                    datasets[aux_dataset] = dataset
        run_majority_vote_xai(xai_predictions, datasets, base_pb)


if __name__ == "__main__":
    # create_xai()

    paths = (
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/transnextupernet_tiny/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/internimageupernet_t/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/vwformermitb3/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/segformerb3/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/vwformerconvnexts/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/fcbformer/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/hardnetdfus/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/segnextl/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/fusegnet/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/unet/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/missformer/k_fold_state.pkl",
        "/home/vab30xh/projects/2025-wound-ambit/out/k_fold_models/hiformerb/k_fold_state.pkl",
    )

    arg_p1 = ['from_kfold']
    arg_p2 = []
    for path in paths:
        arg_p2.extend(['--paths', path])
    arguments = arg_p1 + arg_p2

    # For production
    print("Calling XAI creation with arguments:")
    print(arguments)
    create_xai(arguments)

