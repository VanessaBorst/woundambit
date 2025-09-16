# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

from medseg.config.config import is_hyperopt_run, is_hyperband_run, is_k_fold_run


def resolve_placeholder_path(path: str) -> str:
    if path.startswith("{project_root}"):
        path = path.replace("{project_root}", get_root_path())
    return path


def get_medseg_path() -> str:
    """Returns the path to the python medseg directory. Assumes that this file is located in a direct
    subdirectory of the medseg directory. """
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)
    project_path = os.path.abspath(os.path.join(current_dir_path, ".."))
    return project_path


def get_main_path() -> str:
    """Returns the path to the main directory. """
    src_path = os.path.abspath(os.path.join(get_medseg_path(), ".."))
    return src_path


def get_src_path() -> str:
    """Returns the path to the python src directory. """
    src_path = os.path.abspath(os.path.join(get_main_path(), ".."))
    return src_path


def get_root_path() -> str:
    """Returns the path to the project root directory. """
    root_path = os.path.abspath(os.path.join(get_src_path(), ".."))
    return root_path


class PathBuilder:
    def __init__(self, cfg=None):
        self.path_parts = []
        self.cfg = cfg

    def clone(self) -> 'PathBuilder':
        clone = PathBuilder(self.cfg)
        clone.path_parts = self.path_parts.copy()
        return clone

    @classmethod
    def from_path(cls, path: str, cfg: dict = None) -> 'PathBuilder':
        builder = PathBuilder()
        is_absolute = os.path.isabs(path)
        path_parts = path.split(os.sep)

        # delete empty strings
        path_parts = [part for part in path_parts if part]

        # TODO: This will probably not works under Windows
        if is_absolute:
            path_parts.insert(0, os.sep)  # Restore leading "/"

        builder.path_parts = path_parts
        builder.cfg = cfg
        return builder

    @classmethod
    def trial_out_builder(cls, cfg: dict) -> 'PathBuilder':
        builder = PathBuilder(cfg).root().out()
        if is_hyperopt_run(cfg):
            builder.hyperopt_runs().hyperopt_name()
            if is_hyperband_run(cfg):
                builder.trial_bracket()
        elif is_k_fold_run(cfg):
            builder.k_fold().k_fold_name()
        else:
            builder.single_runs()
        builder.dataset()
        builder.trial_name()
        return builder

    @classmethod
    def hyperopt_out_builder(cls, cfg: dict) -> 'PathBuilder':
        builder = PathBuilder(cfg).root().out()
        if is_hyperopt_run(cfg):
            builder.hyperopt_runs().hyperopt_name()
        return builder

    @classmethod
    def tb_out_builder(cls, cfg: dict) -> 'PathBuilder':
        builder = PathBuilder.trial_out_builder(cfg).tb()
        return builder

    @classmethod
    def img_out_builder(cls, cfg: dict) -> 'PathBuilder':
        return PathBuilder.trial_out_builder(cfg).images()

    @classmethod
    def pretrained_dir_builder(cls) -> 'PathBuilder':
        return PathBuilder().root().add('data').add('pretrained')

    @classmethod
    def out_builder(cls) -> 'PathBuilder':
        return PathBuilder().root().add('out')

    def root(self):
        self.path_parts.append(get_root_path())
        return self

    def out(self):
        self.path_parts.append("out")
        return self

    def single_runs(self):
        self.path_parts.append("single_runs")
        return self

    def trial_name(self):
        self.path_parts.append(self.cfg["trial_name"])
        return self

    def dataset(self):
        self.path_parts.append(self.cfg["dataset"]["type"])
        return self

    def hyperopt_runs(self):
        self.path_parts.append("hyperopt_runs")
        return self

    def hyperopt_name(self):
        if self.cfg is not None and is_hyperopt_run(self.cfg):
            self.path_parts.append(self.cfg["hyperopt_name"])
        return self

    def k_fold(self):
        self.path_parts.append("k_fold")
        return self

    def k_fold_name(self):
        if self.cfg is not None and is_k_fold_run(self.cfg):
            k_fold_name = self.cfg["k_fold"].get("k_fold_name", None)
            if k_fold_name is not None:
                self.path_parts.append(k_fold_name)
        return self

    def k_fold_iter(self):
        if self.cfg is not None and is_k_fold_run(self.cfg):
            k_i = self.cfg["k_fold"].get("k_i", None)
            if k_i is not None:
                self.path_parts.append("k_" + str(k_i).zfill(2))
        return self

    def tb(self):
        # do nothing here as tb path currently is intended to be same as trial path
        return self

    def trial_bracket(self):
        if self.cfg is not None and is_hyperband_run(self.cfg):
            self.path_parts.append("bracket_" + str(self.cfg["trial_bracket"]))
        return self

    def images(self):
        self.path_parts.append("images")
        return self

    def add(self, path_part):
        if path_part is not None:
            self.path_parts.append(path_part)
        return self

    def build(self, make_dirs=True):
        path = os.path.join(*self.path_parts)
        if make_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
