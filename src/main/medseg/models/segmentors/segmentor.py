import inspect
from abc import ABCMeta
from copy import deepcopy
from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.cuda.amp import autocast

from medseg.data.split_type import SplitType
from medseg.util.class_ops import get_class_arg_names
from medseg.util.img_ops import logits_to_segmentation_mask, multiscale


class Segmentor(nn.Module, metaclass=ABCMeta):
    """ Abstract base class for all segmentors. """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.multiscale_cfg = None
        # Store the arguments passed to the constructor
        argspec = inspect.getfullargspec(self.__class__.__init__)
        argnames = argspec.args[1:]  # Skip 'self'
        defaults = argspec.defaults if argspec.defaults else []
        default_args = dict(zip(argnames[-len(defaults):], defaults))

        # Update default args with the passed args and kwargs
        passed_args = {argname: argvalue for argname, argvalue in zip(argnames, args)}
        default_args.update(passed_args)
        default_args.update(kwargs)

        self.init_args = default_args

    @classmethod
    def build_from_cfg(cls, cfg: dict, **kwargs):
        """
        Build a segmentor from a config dict.
        """
        cfg_kwargs = deepcopy(cfg['architecture'])
        allowed_keys = get_class_arg_names(cls)
        if 'kwargs' in allowed_keys:
            # try to remove invalid args
            disallowed_keys = {'arch_type', 'model_name', 'multiscale'}
            allowed_kwargs = {k: v for k, v in cfg_kwargs.items() if k not in disallowed_keys}
        else:
            allowed_kwargs = {k: v for k, v in cfg_kwargs.items() if k in allowed_keys}

        allowed_kwargs.update(kwargs)
        model = cls(**allowed_kwargs)
        model.set_multiscale_cfg(cfg.get('multiscale', None))
        return model

    def default_loss_func(self, multiclass: bool) -> Callable[[Tensor, Tensor], Tensor]:
        if not multiclass:
            print(f"Using default BCEL loss function for binary segmentation")
            return nn.BCEWithLogitsLoss()
        else:
            print(f"Using default CE loss function for multiclass segmentation")
            return nn.CrossEntropyLoss()

    def set_param_groups(self, optimizer: torch.optim.Optimizer):
        """ Sets the parameter groups for the optimizer. """
        pass

    def set_multiscale_cfg(self, multiscale_cfg: Optional[dict]):
        if multiscale_cfg is None: return
        allowed_keys = {'multiscale_factor', 'divisor', 'align_corners'}
        for k in multiscale_cfg.keys():
            if k not in allowed_keys:
                raise ValueError(f"Invalid key {k} in multiscale config. Allowed keys are {allowed_keys}")
        self.multiscale_cfg = multiscale_cfg

    def train_iteration(self, images: Tensor, masks: Tensor, ids: List[int], state):
        # TODO: idea: move this to trainer, for custom function, check if there exists a train_iteration method on the
        # segmentor, if so, call it, otherwise, call the default one
        with autocast(enabled=state.mixed_precision):
            state.current_iteration += 1
            images = images.to(device=state.device, dtype=torch.float)
            masks = masks.to(device=state.device, dtype=torch.long if state.multiclass else torch.float)

            if self.multiscale_cfg is not None:
                images = multiscale(self.in_size, images, masks, **self.multiscale_cfg)
                masks = multiscale(self.in_size, images, masks, **self.multiscale_cfg)
            # forward
            predictions = state.compiled_model(images)
            masks = masks.squeeze(1) if state.multiclass else masks
            loss = state.loss_func(predictions, masks)

            # backward
            state.optimizer.zero_grad()
            state.scaler.scale(loss).backward()
            state.scaler.step(state.optimizer)
            state.scaler.update()
            state.train_loss_tracker.update(loss)
            # metrics
            predictions = logits_to_segmentation_mask(predictions).long() if state.multiclass else predictions > 0.5
            img_ids = [state.dataset_manager.get_train_dataset().get_image_file_name(real_i) for real_i in ids]
            train_metrics_tracker = state.metrics_manager.get_last_tracker(SplitType.TRAIN)
            train_metrics_tracker.update_metrics_from_batch(img_ids, predictions.cpu(), masks.cpu())


