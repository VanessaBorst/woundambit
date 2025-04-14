import torchvision

torchvision.disable_beta_transforms_warning()
import pickle

import click
import torch
import os

from medseg.config.check_config import check_config
from medseg.config.config import load_and_parse_config, is_hyperopt_run, is_hyperband_run, is_k_fold_run
from medseg.hyperopt.hyperband import HyperbandOptimizer
from medseg.hyperopt.grid_search import GridSearchOptimizer
from medseg.training.k_fold import KFoldCrossValidator
from medseg.training.trainer import Trainer
from medseg.util.date_time import get_current_date_time_str

@click.group()
def train():
    pass


@train.command(name='from_config')
@click.option("--path", type=click.Path(exists=True, file_okay=True), required=True)
@click.option("--simulate", type=bool, is_flag=True, default=False, required=False)
def train_from_config(path: str, simulate: bool = False):
    print(f"Using config file under: {path}")
    # print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    cfg = load_and_parse_config(path)
    cfg['trial_name'] = f"{cfg['architecture']['model_name']}_{get_current_date_time_str()}"
    check_config(cfg)
    if is_hyperopt_run(cfg):
        hyperopt_type = cfg['hyperopt']['type'].lower()
        if hyperopt_type == 'hyperband':
            hb_opt = HyperbandOptimizer(cfg, cfg_path=path, simulate=simulate)
            hb_opt.run()
        elif hyperopt_type == 'grid_search':
            grid_opt = GridSearchOptimizer(cfg, cfg_path=path, simulate=simulate)
            grid_opt.run()
        else:
            raise ValueError(f"Unknown hyperopt type: {hyperopt_type}")
    elif is_k_fold_run(cfg):
        kfold = KFoldCrossValidator(cfg, cfg_or_dir_path=path)
        kfold.run()
    else:
        if simulate:
            print(
                "Simulation mode is currently only supported for hyperparameter optimization runs. Ignoring setting...")
        trainer = Trainer(cfg, cfg_path=path)
        trainer.train()


@train.command(name='from_checkpoint')
@click.option("--path", type=click.Path(exists=True, dir_okay=True), required=True)
@click.option("--add_epochs", type=int, default=0, required=False)
def train_from_checkpoint(path: str, add_epochs: int = 0):
    print(f"Resuming checkpoint from: {path}")
    checkpoint = torch.load(path)
    cur_epochs = checkpoint['current_epoch']
    cur_iters = checkpoint['current_iteration']
    print(f"Checkpoint was stopped after {cur_epochs} epochs (={cur_iters} iterations).")
    if add_epochs > 0:
        max_epochs = cur_epochs + add_epochs
        print(f"Setting training run to terminate after {cur_epochs} + {add_epochs} = {max_epochs} epochs.")
        checkpoint['cfg']['settings']['max_epochs'] = checkpoint['current_epoch'] + add_epochs
    trainer = Trainer.from_state_dict(checkpoint)
    trainer.train()


@train.command(name='from_hyperopt_state')
@click.option("--path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--grid_search_active", type=bool, is_flag=True, default=False, required=False)
@click.option("--simulate", type=bool, is_flag=True, default=False, required=False)
def train_from_hyperopt_state(path: str, grid_search_active: bool = False, simulate: bool = False):
    print(f"Trying to resume hyperparameter optimization run from: {path}")
    if grid_search_active:
        gs_optimizer = GridSearchOptimizer.resume_from_path(path, simulate=simulate)
        gs_optimizer.run()
    else:
        state = pickle.load(open(path, 'rb'))
        cfg = state['cfg']
        if is_hyperband_run(cfg):
            cfg['hyperopt_name'] = cfg['hyperopt_name'] + "_simulate" if simulate else cfg['hyperopt_name']
            hb_optimizer = HyperbandOptimizer.from_state_dict(state, simulate=simulate)
            hb_optimizer.run()
        else:
            # Resume not yet possible
            print("Resume function is only implemented for Hyperband" if is_hyperopt_run(cfg)
                  else "Not a hyperparameter optimization run.")


@train.command(name='from_kfold_state')
@click.option("--path", type=click.Path(exists=True, dir_okay=False), required=True)
def train_from_kfold_state(path: str, simulate: bool = False):
    print(f"Trying to resume k-fold cross-validation run from: {path}")
    state = pickle.load(open(path, 'rb'))
    cfg = state['cfg']

    if is_k_fold_run(cfg):
        hb_optimizer = KFoldCrossValidator.from_state_dict(state, path)
        hb_optimizer.run()


if __name__ == "__main__":
    train()
