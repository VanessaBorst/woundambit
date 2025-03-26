import torchvision

from medseg.evaluation.metrics import EvalMetric
from medseg.evaluation.metrics_manager import MetricsManager
from medseg.evaluation.plots import save_boxplot
from medseg.util.path_builder import PathBuilder

torchvision.disable_beta_transforms_warning()

import os
import pickle
from typing import Optional, Tuple, List, Dict

import click
import torch

from medseg.data.split_type import SplitType
from medseg.evaluation.ensemble_evaluator import EnsembleEvaluator
from medseg.evaluation.evaluator import Evaluator


@click.group()
def evaluate():
    pass


@evaluate.command(name='from_checkpoint')
@click.option('--path', type=str, required=True)
@click.option('--split', type=str, required=True)
@click.option('--for_epochs', type=int, required=False)
@click.option('--eval_object_sizes', type=bool, is_flag=True, default=False, required=False)
@click.option('--save_samples', type=bool, is_flag=True, default=False, required=False)
@click.option('--use_xai', is_flag=True, default=False, required=False)
def evaluate_from_checkpoint(path: str, split: str, for_epochs: Optional[int] = None, eval_object_sizes: bool = False,
                             save_samples: bool = False, use_xai: bool = False):
    split = SplitType(split.lower())
    checkpoint = torch.load(path)
    checkpoint['cfg']['settings']['eval_object_sizes'] = eval_object_sizes
    checkpoint['cfg']['settings']['save_sample_segmentations'] = save_samples
    evaluator = Evaluator.from_checkpoint(checkpoint, path, split, use_xai)
    evaluator.evaluate(training_epoch=0, for_epochs=for_epochs, is_final_eval=True)



@evaluate.command(name='from_kfold')
@click.option('--path', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--add_aux_test_set', type=str, required=False, multiple=True)
@click.option("--use_xai", type=bool, is_flag=True, default=False, required=False)
def evaluate_from_kfold(path: str, add_aux_test_set: Optional[Tuple[str]] = None, use_xai: bool = False):
    state = pickle.load(open(path, 'rb'))
    path = os.path.dirname(path)
    assert os.path.isabs(path), f"Path must be absolute, but is: {path}"
    if add_aux_test_set is not None:
        add_aux_test_set = list(add_aux_test_set)
    evaluator = EnsembleEvaluator.from_kfold_state(state, path, add_aux_test_set, use_xai)
    evaluator.run_evaluation()


@evaluate.command(name='create_boxplot')
@click.option('--paths', type=str, required=True, multiple=True)
@click.option('--split', type=str, required=True)
@click.option('--metric', type=str, required=True)
@click.option('--dataset', type=str, required=True)
@click.option('--save_dir', type=str, required=True)
@click.option('--model_names', type=str, required=False, multiple=True)
def create_boxplot(paths: List[str], split: str, metric: str, dataset: str, save_dir: str,
                   model_names: Optional[List[str]] = None):
    for path in paths: assert os.path.exists(path), f"Path does not exist: {path}"
    split = SplitType(split.lower())
    metric = EvalMetric(metric.lower())
    model_metrics_dict: Dict[str, List] = dict()
    for i, path in enumerate(paths):
        checkpoint = torch.load(path)
        evaluator = Evaluator.from_checkpoint(checkpoint, path, split)
        metrics_managers = evaluator.evaluate(training_epoch=0, for_epochs=None, is_final_eval=True)
        mm: MetricsManager = metrics_managers[dataset.lower()]
        tracker = mm.get_last_tracker(split)
        assert tracker is not None
        metric_list = tracker.get_per_sample_metric_list(metric)
        model_name = checkpoint['cfg']['architecture']['model_name'] if model_names is None else model_names[i]
        assert model_name not in model_metrics_dict.keys(), f"Model name already exists: {model_name}"
        model_metrics_dict[model_name] = metric_list
    save_path = PathBuilder.from_path(save_dir).add(f"boxplot_{dataset}_{split.value}_{metric.value()}.png").build()
    save_boxplot(model_metrics_dict, save_path, metric.get_abbr())


@evaluate.command(name='create_boxplot_from_kfold')
@click.option('--paths', type=str, required=True, multiple=True)
@click.option('--metric', type=str, required=True)
@click.option('--save_dir', type=str, required=True)
@click.option('--model_names', type=str, required=False, multiple=True)
def create_boxplot_from_kfold(paths: List[str], metric: str, save_dir: str, model_names: Optional[List[str]] = None):
    for path in paths: assert os.path.exists(path), f"Path does not exist: {path}"
    metric = EvalMetric(metric.lower())
    ds_model_metrics_dict_ens_avg: Dict[str, Dict[str, List]] = dict()
    ds_model_metrics_dict_ens_mv: Dict[str, Dict[str, List]] = dict()
    for i, path in enumerate(paths):
        state = pickle.load(open(path, 'rb'))
        path = os.path.dirname(path)
        assert os.path.isabs(path), f"Path must be absolute, but is: {path}"
        evaluator = EnsembleEvaluator.from_kfold_state(state, path)
        trackers_ens_avg, trackers_ens_mv = evaluator.run_evaluation()
        model_name = state['cfg']['architecture']['model_name'] if model_names is None else model_names[i]

        for n, t in trackers_ens_avg.items():
            if n not in ds_model_metrics_dict_ens_avg.keys():
                ds_model_metrics_dict_ens_avg[n] = dict()
            ds_model_metrics_dict_ens_avg[n][model_name] = t.get_per_sample_metric_list(metric)
        for n, t in trackers_ens_mv.items():
            if n not in ds_model_metrics_dict_ens_mv.keys():
                ds_model_metrics_dict_ens_mv[n] = dict()
            ds_model_metrics_dict_ens_mv[n][model_name] = t.get_per_sample_metric_list(metric)

    # loop through models and datasets and save boxplots
    for ds_name, model_metrics_dict in ds_model_metrics_dict_ens_avg.items():
        save_path = PathBuilder.from_path(save_dir).add(
            f"boxplot_ens_avg_{ds_name}_{metric.value()}.png").build()
        save_boxplot(model_metrics_dict, save_path, metric.get_abbr())

    for ds_name, model_metrics_dict in ds_model_metrics_dict_ens_mv.items():
        save_path = PathBuilder.from_path(save_dir).add(
            f"boxplot_ens_mv_{ds_name}_{metric.value()}.png").build()
        save_boxplot(model_metrics_dict, save_path, metric.get_abbr())


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-e5cc7cc7-0a84-5237-9062-30b65511cb40"
    evaluate()
