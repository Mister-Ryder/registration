"""Authoritative DDP runtime using the faithful first-stage objective."""

from __future__ import annotations

from torch.nn.parallel import DistributedDataParallel

from . import training_engine as _engine
from .network import V4FinalRegistrationModel, faithful_representation_objective


_raw_validation = _engine._validation_mean


def _safe_validation(model, *args, **kwargs):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return _raw_validation(model, *args, **kwargs)


_engine.V4FinalModel = V4FinalRegistrationModel
_engine.representation_objective = faithful_representation_objective
_engine._validation_mean = _safe_validation
train_descriptor = _engine.train_descriptor


__all__ = ["train_descriptor"]

