"""Auditable native 3-D PRA-CM train/validation engine."""

from __future__ import annotations

import json
import math
import shutil
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data import L2RMRCTDataset, PLCRVolumeDataset, single_case_collate
from ..ops.spatial import jacobian_determinant, warp
from .checkpoint import (
    atomic_torch_save,
    checkpoint_payload,
    config_sha256,
    load_checkpoint,
    restore_rng_state,
    source_tree_sha256,
)
from .module_v4 import PRACMTrainingModule


def _ensure_checkpoint_space(output: Path, minimum_free_gib: float) -> None:
    free_gib = shutil.disk_usage(output).free / (1024.0 ** 3)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"Refusing a new checkpoint with only {free_gib:.2f} GiB free; "
            f"the frozen safety floor is {minimum_free_gib:.2f} GiB."
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _average(rows):
    result: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            result[key] = result.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: value / counts[key] for key, value in result.items()}


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1)


def _dice(fixed_label: torch.Tensor, warped_moving_label: torch.Tensor) -> torch.Tensor:
    fixed = fixed_label.bool()
    moving = warped_moving_label > 0.5
    intersection = (fixed & moving).sum(dtype=torch.float32)
    denominator = fixed.sum(dtype=torch.float32) + moving.sum(dtype=torch.float32)
    return (2 * intersection + 1.0e-6) / (denominator + 1.0e-6)


def _validation_edges(batch, result):
    if batch["kind"] == "pair":
        return (
            (
                result.registrations[0],
                batch["fixed_domain"],
                batch.get("fixed_label"),
                batch.get("moving_label"),
            ),
        )
    if batch["kind"] == "unpaired_domains":
        # Synthetic fixed-domain masks are already folded into endpoint_valid.
        # Use the whole tensor grid as the denominator so support loss remains
        # visible instead of becoming one by construction.
        return tuple(
            (output, torch.ones_like(output.endpoint_valid), None, None)
            for output in result.registrations
        )
    domains = batch["domains"]
    labels = batch.get("labels", (None, None, None))
    return tuple(
        (result.registrations[output_index], domains[fixed_index], labels[fixed_index], labels[moving_index])
        for output_index, (fixed_index, moving_index) in enumerate(((0, 1), (1, 2), (0, 2)))
    )


@torch.no_grad()
def validate(module, loader, device, precision, limit: int):
    module.eval()
    module.set_validation_mode(True)
    rows = []
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        batch = _to_device(batch, device)
        is_unpaired = batch["kind"] == "unpaired_domains"
        if is_unpaired:
            # Validation synthetic fields must be identical across epochs.
            # fork_rng prevents validation from perturbing the training RNG.
            cuda_devices = []
            if device.type == "cuda":
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=cuda_devices):
                augmentation_seed = int(batch["augmentation_seed"])
                torch.manual_seed(augmentation_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(augmentation_seed)
                with _autocast(device, precision):
                    result = module(batch, include_synthetic=True)
        else:
            with _autocast(device, precision):
                result = module(batch, include_synthetic=False)
        row = {name: value.detach().float().item() for name, value in result.losses.items()}
        edge_rows = []
        for output, fixed_domain, fixed_label, moving_label in _validation_edges(batch, result):
            fixed_support = fixed_domain.bool()
            endpoint = output.endpoint_valid.bool() & fixed_support
            determinant = jacobian_determinant(output.flow.float())
            edge = {
                "effective_support": _masked_mean(
                    output.endpoint_valid.float(), fixed_support
                ).item(),
                "fold_fraction": _masked_mean(
                    (determinant <= 0).float(), fixed_support
                ).item(),
                "mean_entropy": _masked_mean(output.entropy.float(), endpoint).item(),
            }
            if fixed_label is not None and moving_label is not None:
                warped_label = warp(
                    moving_label.float(),
                    output.flow.float(),
                    mode="nearest",
                    padding_mode="zeros",
                )
                edge["label_dsc"] = _dice(fixed_label, warped_label).item()
            edge_rows.append(edge)
        row.update(_average(edge_rows))
        rows.append(row)
    if not rows:
        raise RuntimeError("Validation loader produced no cases.")
    module.set_validation_mode(False)
    return _average(rows)


def _selection_score(metrics: Mapping[str, float], use_labels: bool) -> float:
    if not use_labels:
        return float(metrics["selection_total"])
    if "label_dsc" not in metrics:
        raise RuntimeError("Validation-label checkpoint selection was enabled but no DSC was computed.")
    return (
        1.0
        - float(metrics["label_dsc"])
        + 0.50 * float(metrics["fold_fraction"])
        + 0.10 * (1.0 - float(metrics["effective_support"]))
    )


def _learning_rate_factor(config: ExperimentConfig, epoch_index: int) -> float:
    warmup = config.training.warmup_epochs
    floor = config.training.minimum_learning_rate_factor
    if warmup and epoch_index < warmup:
        return (epoch_index + 1) / warmup
    decay_epochs = max(1, config.training.epochs - warmup - 1)
    progress = min(1.0, max(0.0, (epoch_index - warmup) / decay_epochs))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_training(
    config: ExperimentConfig,
    *,
    output_dir: Union[str, Path],
    tensorboard_dir: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    resume: Optional[Union[str, Path]] = None,
    max_epochs_this_invocation: Optional[int] = None,
) -> Mapping[str, object]:
    if max_epochs_this_invocation is not None and max_epochs_this_invocation < 1:
        raise ValueError("max_epochs_this_invocation must be positive when provided.")
    output = Path(output_dir).expanduser().resolve()
    tensorboard_path = (
        Path(tensorboard_dir).expanduser().resolve()
        if tensorboard_dir is not None
        else output / "tf-logs"
    )
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty training directory without --resume: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    _seed_everything(config.training.seed)
    if config.data.dataset == "l2r_mrct":
        if config.training.validation_label_selection:
            raise ValueError(
                "L2R public labels are evaluation-only; set validation_label_selection=false."
            )
        train_dataset = L2RMRCTDataset(
            config.data, split="train", seed=config.training.seed
        )
        validation_dataset = L2RMRCTDataset(
            config.data,
            split="validation",
            samples=config.training.validation_cases,
            seed=config.training.seed + 17,
        )
    else:
        train_dataset = PLCRVolumeDataset(
            config.data,
            split="train",
            seed=config.training.seed,
            load_labels=False,
        )
        validation_dataset = PLCRVolumeDataset(
            config.data,
            split="validation",
            samples=config.training.validation_cases,
            seed=config.training.seed + 17,
            load_labels=config.training.validation_label_selection,
            pair_only=True,
        )
    data_identity = {
        "train": train_dataset.source_identity(),
        "validation": validation_dataset.source_identity(),
    }
    loader_kwargs = dict(
        batch_size=1,
        num_workers=config.training.workers,
        pin_memory=device_object.type == "cuda",
        collate_fn=single_case_collate,
    )
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    module = PRACMTrainingModule(
        config.model, config.losses, config.augmentation, config.training
    ).to(device_object)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: _learning_rate_factor(config, epoch)
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=device_object.type == "cuda" and config.training.precision == "fp16"
    )
    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    early_stopping_counter = 0
    config_dict = config.to_dict()
    if resume is not None:
        # RNG snapshots are CPU byte tensors.  Loading an entire checkpoint
        # directly onto CUDA moves them as well and makes torch.set_rng_state
        # fail during an otherwise valid resume.  Model/optimizer loaders move
        # their own tensors to the parameter device after this CPU load.
        payload = load_checkpoint(resume, map_location="cpu")
        if payload["config_sha256"] != config_sha256(config_dict):
            raise ValueError("Resume configuration differs from the checkpoint.")
        if payload["source_sha256"] != source_tree_sha256():
            raise ValueError("Resume source tree differs from the checkpoint.")
        if payload["data_identity"] != data_identity:
            raise ValueError("Resume cohort, inventory, phase protocol, or label policy differs.")
        module.load_state_dict(payload["module_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_validation = float(payload.get("best_validation", best_validation))
        early_stopping_counter = int(payload.get("early_stopping_counter", 0))

    manifest = {
        "schema": "pracm_v4_3d_training_run_v1",
        "protocol_id": config.training.protocol_id,
        "source_sha256": source_tree_sha256(),
        "config_sha256": config_sha256(config_dict),
        "config": config_dict,
        "data_identity": data_identity,
        "training_uses_segmentation_labels": False,
        "storage_policy": {
            "minimum_free_disk_gib_before_checkpoint": config.training.minimum_free_disk_gib,
            "core_artifacts_only": True,
        },
        "tensorboard_dir": str(tensorboard_path),
        "early_stopping_applied": config.training.early_stopping_enabled,
        "early_stopping_policy": (
            "minimum_epochs_and_patience"
            if config.training.early_stopping_enabled
            else f"{config.training.protocol_id}_fixed_budget_diagnostic_only"
        ),
        "validation_labels": (
            "liver masks used only for checkpoint selection"
            if config.training.validation_label_selection
            else "not used"
        ),
        "checkpoint_selection": (
            "1-DSC + 0.5*fold_fraction + 0.1*(1-effective_support)"
            if config.training.validation_label_selection
            else "validation training objective"
        ),
        "pairwise_inference": True,
        "status": "running",
    }
    manifest_path = output / "run_manifest.json"
    if resume is not None and manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("tensorboard_dir") != str(tensorboard_path):
            raise ValueError("Resume TensorBoard directory differs from the formal run manifest.")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    metrics_path = output / "epoch_metrics.jsonl"
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("PRA-CM formal training requires the tensorboard package.") from error
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_path), max_queue=1000, flush_secs=120)
    completed_epochs = start_epoch
    stopped_early = (
        config.training.early_stopping_enabled
        and
        start_epoch >= config.training.minimum_epochs
        and early_stopping_counter >= config.training.early_stopping_patience
    )
    invocation_stop = config.training.epochs
    if max_epochs_this_invocation is not None:
        invocation_stop = min(
            invocation_stop, start_epoch + int(max_epochs_this_invocation)
        )
    epoch_range = () if stopped_early else range(start_epoch, invocation_stop)
    for epoch in epoch_range:
        if device_object.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_object)
        train_dataset.set_epoch(epoch)
        module.set_validation_mode(False)
        module.train()
        rows = []
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        for batch in train_loader:
            module.set_training_state(epoch, global_step)
            batch = _to_device(batch, device_object)
            optimizer.zero_grad(set_to_none=True)
            # Pair steps add exact synthetic correspondence; relational steps
            # retain the three real edges of one selected P/Ci/Cj triangle.
            include_synthetic = (
                batch["kind"] == "unpaired_domains"
                or (
                    batch["kind"] == "pair"
                    and global_step % config.training.synthetic_every_steps == 0
                )
            )
            with _autocast(device_object, config.training.precision):
                result = module(batch, include_synthetic=include_synthetic)
            if not torch.isfinite(result.loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={global_step}.")
            scaler.scale(result.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(module.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            rows.append({name: value.detach().float().item() for name, value in result.losses.items()})
            global_step += 1
            for name, value in rows[-1].items():
                writer.add_scalar(f"iteration/{name}", value, global_step)
            writer.add_scalar("iteration/learning_rate", epoch_learning_rate, global_step)
            writer.add_scalar("iteration/logical_step", global_step, global_step)
            writer.add_scalar("iteration/optimizer_step", global_step, global_step)
            if global_step % config.training.log_every_steps == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "logical_step": global_step,
                            "optimizer_step": global_step,
                            "kind": batch["kind"],
                            "patient_id": batch.get("patient_id"),
                            "total": rows[-1]["total"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        train_metrics = _average(rows)
        validation_metrics = validate(
            module,
            validation_loader,
            device_object,
            config.training.precision,
            config.training.validation_cases,
        )
        selection_score = _selection_score(
            validation_metrics, config.training.validation_label_selection
        )
        improved = selection_score < best_validation
        if improved:
            best_validation = selection_score
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        scheduler.step()
        _ensure_checkpoint_space(output, config.training.minimum_free_disk_gib)
        payload = checkpoint_payload(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config_dict,
            epoch=epoch,
            global_step=global_step,
            best_validation=best_validation,
            data_identity=data_identity,
            early_stopping_counter=early_stopping_counter,
        )
        atomic_torch_save(payload, checkpoints / "last.pt")
        if improved:
            atomic_torch_save(payload, checkpoints / "best.pt")
        if (
            (epoch + 1) % config.training.checkpoint_every_epochs == 0
            and (epoch + 1) < config.training.epochs
        ):
            atomic_torch_save(
                payload, checkpoints / f"epoch_{epoch + 1:04d}.pt"
            )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "logical_step": global_step,
            "optimizer_step": global_step,
            "learning_rate_used": epoch_learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_score": selection_score,
            "best_selection_score": best_validation,
            "early_stopping_counter": early_stopping_counter,
            "would_trigger_early_stopping": (
                (epoch + 1) >= config.training.minimum_epochs
                and early_stopping_counter >= config.training.early_stopping_patience
            ),
        }
        if device_object.type == "cuda":
            record["gpu_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated(device_object) / (1024.0**2)
            )
            record["gpu_peak_reserved_mib"] = (
                torch.cuda.max_memory_reserved(device_object) / (1024.0**2)
            )
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        for name, value in train_metrics.items():
            writer.add_scalar(f"train/{name}", value, epoch)
        for name, value in validation_metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        writer.add_scalar("optimization/learning_rate", epoch_learning_rate, epoch)
        writer.add_scalar("optimization/logical_step", global_step, epoch)
        writer.add_scalar("optimization/optimizer_step", global_step, epoch)
        writer.add_scalar("selection/score", selection_score, epoch)
        writer.add_scalar("selection/best_score", best_validation, epoch)
        if device_object.type == "cuda":
            writer.add_scalar(
                "resources/gpu_peak_allocated_mib",
                record["gpu_peak_allocated_mib"],
                epoch,
            )
            writer.add_scalar(
                "resources/gpu_peak_reserved_mib",
                record["gpu_peak_reserved_mib"],
                epoch,
            )
        writer.flush()
        completed_epochs = epoch + 1
        if (
            config.training.early_stopping_enabled
            and
            completed_epochs >= config.training.minimum_epochs
            and early_stopping_counter >= config.training.early_stopping_patience
        ):
            stopped_early = True
            break

    writer.close()

    training_complete = stopped_early or completed_epochs >= config.training.epochs
    manifest.update(
        {
            "status": "completed" if training_complete else "paused",
            "completed_epochs": completed_epochs,
            "global_step": global_step,
            "logical_step": global_step,
            "optimizer_step": global_step,
            "best_selection_score": best_validation,
            "stopped_early": stopped_early,
            "training_complete": training_complete,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "epochs": completed_epochs,
        "global_step": global_step,
        "logical_step": global_step,
        "optimizer_step": global_step,
        "best_selection_score": best_validation,
        "stopped_early": stopped_early,
        "training_complete": training_complete,
    }
