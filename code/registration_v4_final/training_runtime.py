"""Public training entry with uneven-validation-safe DDP dispatch."""

from __future__ import annotations

from torch.nn.parallel import DistributedDataParallel

from . import training_engine as _engine


_original_validation_mean = _engine._validation_mean


def _validation_without_ddp_forward_collectives(model, *args, **kwargs):
    # Validation anchors need not divide evenly across ranks.  Calling the DDP
    # wrapper itself would broadcast buffers on an unequal number of forwards.
    # Gradients are disabled, so evaluating the already synchronized raw module
    # is both correct and deadlock-free.
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return _original_validation_mean(model, *args, **kwargs)


_engine._validation_mean = _validation_without_ddp_forward_collectives
train_descriptor = _engine.train_descriptor


__all__ = ["train_descriptor"]

