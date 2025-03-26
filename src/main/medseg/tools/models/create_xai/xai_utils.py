import gc
from tqdm import tqdm
import torch, os
import numpy as np
from medseg.evaluation.medcam.medcam_utils import save_attention_map
from medseg.evaluation.medcam import medcam

from typing import List
from copy import deepcopy
from medseg.util.path_builder import PathBuilder
from medseg.data.dataset_manager import DatasetManager
from medseg.models.model_builder import build_model
from medseg.data.split_type import SplitType


def _load_model_from_checkpoint(checkpoint, device: torch.device, cfg: dict = None):
    if cfg is not None:
        checkpoint = torch.load(checkpoint, map_location="cpu")
        checkpoint['cfg'] = cfg
    assert "cfg" in checkpoint, "No cfg found in checkpoint"
    cfg = checkpoint['cfg']
    dataset_manager = DatasetManager(cfg)
    class_mapping = None
    for dataset in dataset_manager.datasets.values():
        class_mapping = dataset.class_mapping if class_mapping is None else class_mapping
    assert class_mapping is not None, "No class mapping found in dataset manager"
    out_channels = class_mapping.num_classes if class_mapping.multiclass else 1
    print("Building model. This may take a while...")
    model = build_model(cfg, out_channels=out_channels)
    print("Model built")
    model.to(device)
    model.load_state_dict(checkpoint['model'])

    return model, dataset_manager, cfg["dataset"]["type"]


def _load_kfold_model_from_state(state):
    cfg = state['cfg']
    trained_model_paths = cfg["k_fold"]["trained_model_paths"]
    for trained_model_path in trained_model_paths:
        assert os.path.exists(trained_model_path), f"Checkpoint path {trained_model_path} does not exist"
    main_eval_split = SplitType.TEST if cfg['k_fold'].get('include_test_split', False) else SplitType.VAL
    return trained_model_paths, main_eval_split


def _check_and_prepare_cfgs(checkpoints, additional_aux_datasets):
    def _fetch_cfgs(checkpoints) -> List[dict]:
        # load all checkpoints and only fetch the cfg
        cfgs = list()
        for path in checkpoints:
            checkpoint_dict = torch.load(path, map_location=torch.device('cpu'))
            cfgs.append(deepcopy(checkpoint_dict['cfg']))
            del checkpoint_dict
            torch.cuda.empty_cache()
            gc.collect()
        return cfgs

    def print_warning(value_name: str, value: any):
        print(f"Ensemble evaluation: Difference in '{value_name}' of the given checkpoints detected.")
        print(f"Using {value_name} {str(value)} of the first checkpoint for all checkpoints.")

    cfgs = _fetch_cfgs(checkpoints)
    test_transforms = [cfg['transforms']['test'] for cfg in cfgs]
    # check equality of all transforms
    if not all([test_transforms[0] == test_transforms[i] for i in range(1, len(test_transforms))]):
        print_warning("test transform pipeline", '')
        for i in range(1, len(test_transforms)): cfgs[i]['transforms']['test'] = test_transforms[0]

    random_seeds = [cfg['settings']['random_seed'] for cfg in cfgs]
    if not all([random_seeds[0] == random_seeds[i] for i in range(1, len(random_seeds))]):
        print_warning("random seed", random_seeds[0])
        for i in range(1, len(random_seeds)):
            cfgs[i]['settings']['random_seed'] = random_seeds[0]

    dataset_types = [cfg['dataset']['type'] for cfg in cfgs]
    if not all([dataset_types[0] == dataset_types[i] for i in range(1, len(dataset_types))]):
        print("Ensemble evaluation: Difference in dataset types of the given checkpoints detected.")
        print(f"Using the dataset type {dataset_types[0]} of the first checkpoint for all checkpoints.")
        print_warning("dataset type", dataset_types[0])
        for i in range(1, len(dataset_types)):
            cfgs[i]['dataset']['type'] = dataset_types[0]

    aux_datasets = [cfg['dataset'].get('aux_test_datasets', None) for cfg in cfgs]
    if not all([aux_datasets[0] == aux_datasets[i] for i in range(1, len(aux_datasets))]):
        print_warning("auxiliary test datasets", aux_datasets[0])
        for i in range(1, len(aux_datasets)):
            cfgs[i]['dataset']['aux_test_datasets'] = aux_datasets[0]

    # set default settings
    model_name = cfgs[0]['architecture']['model_name']
    for cfg in cfgs:
        cfg['settings']['eval_object_sizes'] = True
        cfg['settings']['num_workers'] = 1
        cfg['settings']['batch_size'] = 1
        cfg['settings']['final_eval_epochs'] = 1
        if additional_aux_datasets is not None and len(additional_aux_datasets) > 0:
            aux_datasets = cfg['dataset'].get('aux_test_datasets', list())
            aux_datasets.extend(additional_aux_datasets)
            cfg['dataset']['aux_test_datasets'] = aux_datasets
    return cfgs, model_name


def prepare_normal_cfg(cfg):
    cfg['settings']['eval_object_sizes'] = True
    cfg['settings']['num_workers'] = 1
    cfg['settings']['batch_size'] = 1
    cfg['settings']['final_eval_epochs'] = 1
    return cfg


def create_path_builder() -> PathBuilder:
    return PathBuilder().root().add(f"out/xai_creation/")


def create_xai_images(model, dataset_manager, split, base_pb, real_num_img, xai_prediction_hook=None, dataset_name=None,
                      aux=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if aux:
        dataset, loader = dataset_manager
    elif dataset_manager.has_split(split):
        dataset, loader = dataset_manager.get_dataset_and_loader(split)
    else:
        raise ValueError(f"Dataset manager does not have split {split}")
    model = medcam.inject(model, output_dir=base_pb.clone().build(),
                          backend='gcam', data_shape=(512,512),
                          save_maps=True)

    model.eval()
    for i, ([images, masks], ids) in tqdm(enumerate(loader), desc="Images", total=real_num_img):
        img_filenames = [dataset.get_image_file_name(real_i) for real_i in ids]
        with torch.no_grad():
            images = images.to(device=device, dtype=torch.float)
            _ = model(images, img_names=img_filenames, xai_predictions_hook=xai_prediction_hook,
                      dataset_name=dataset_name, is_aux_test_data=aux)
        if i >= real_num_img:
            break
    return dataset


def xai_mean_vote(mask_preds: list[np.ndarray]) -> np.ndarray:
    mask_preds = np.array(mask_preds)
    mean_mask = np.mean(mask_preds, axis=0)
    return np.squeeze(mean_mask)


def run_majority_vote_xai(xai_predictions, datasets, save_pb: PathBuilder):
    for dataset_name, layers in xai_predictions.items():
        for layer, models_preds in layers.items():
            n_models = len(models_preds)
            assert n_models > 1, "Ensemble evaluation: Majority vote evaluation requires at least two models."
            assert all([models_preds[0].keys() == models_preds[i].keys() for i in range(1, n_models)]), \
                f"Ensemble evaluation: XAI predictions for {dataset_name} must have the same prediction filenames."
            dataset = datasets[dataset_name]
            img_keys = list(models_preds[0].keys())
            for img_key in img_keys:
                xai_preds = [models_preds[i][img_key] for i in range(n_models)]
                xai_preds_majority = xai_mean_vote(xai_preds)
                real_i = dataset.all_images.index(img_key)
                ds_i = dataset.real_index_to_dataset_index(real_i)
                image, mask, _ = dataset.__getitem__(ds_i)
                pred = torch.from_numpy(xai_preds_majority)
                save_path = save_pb.clone().add(f"Aux-{dataset_name}_mean_vote").add(img_key).build()
                # save_pb.clone().add(dataset_name).add(f"aux-{dataset_name}").add(f"aux_xai_eval_{dataset_name}_{dataset.split_type.value}_{layer}_mean_vote").add(img_key).build()
                save_attention_map(filename=save_path, attention_map=pred, heatmap=True, raw_input=image)