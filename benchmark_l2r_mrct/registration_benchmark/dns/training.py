"""Self-supervised MASR-Net pretraining on standalone unlabeled volumes."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..contract import canonical_json_hash, load_registration_manifest
from ..io import load_scalar, robust_unit_interval
from .augmentation import stochastic_nonlinear_transform
from .model import MASRNet
from ..provenance import sha256_file
from ..tensorboard_logging import create_summary_writer, log_iteration, log_record
from ..checkpointing import save_training_checkpoints


def _write_run_lock(path: Path, payload: Dict[str, Any]) -> None:
    """Create an immutable protocol lock without depending on another trainer."""

    payload = dict(payload)
    payload["content_sha256"] = canonical_json_hash(payload)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"RUN_LOCK mismatch: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def _convergence_diagnostic(history) -> Dict[str, Any]:
    window = [float(value) for value in history[-20:]]
    if len(window) < 20:
        return {"available": False, "window_epochs": len(window), "required_epochs": 20}
    slope = float(np.polyfit(np.arange(20, dtype=np.float64), np.asarray(window), 1)[0])
    scale = max(abs(float(np.mean(window))), 1e-8)
    return {
        "available": True, "window_epochs": 20, "validation_loss_slope_per_epoch": slope,
        "relative_slope_per_epoch": slope / scale,
        "not_converged_still_improving": bool(slope / scale < -1e-4),
    }


def _crop_or_pad(data_xyz: np.ndarray, shape_zyx: Sequence[int], rng: random.Random) -> torch.Tensor:
    data = torch.from_numpy(np.asarray(data_xyz.transpose(2, 1, 0), dtype=np.float32))[None, None]
    target = tuple(int(v) for v in shape_zyx)
    pads = []
    for current, wanted in zip(reversed(data.shape[-3:]), reversed(target)):
        total = max(wanted - current, 0)
        pads.extend([total // 2, total - total // 2])
    if any(pads):
        data = F.pad(data, tuple(pads), mode="replicate")
    slices = []
    for current, wanted in zip(data.shape[-3:], target):
        start = rng.randint(0, current - wanted) if current > wanted else 0
        slices.append(slice(start, start + wanted))
    return data[..., slices[0], slices[1], slices[2]]


def anatomy_contrastive_loss(
    descriptor: torch.Tensor,
    augmented_descriptor: torch.Tensor,
    foreground: torch.Tensor,
    *,
    n_samples: int = 8196,
    temperature: float = 0.07,
    chunk_size: int = 512,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if descriptor.shape != augmented_descriptor.shape:
        raise ValueError("Descriptor tensors must have identical shapes.")
    locations = torch.nonzero(foreground[0, 0].reshape(-1), as_tuple=False).reshape(-1)
    if locations.numel() < 2:
        locations = torch.arange(descriptor.shape[-3] * descriptor.shape[-2] * descriptor.shape[-1], device=descriptor.device)
    count = min(int(n_samples), int(locations.numel()))
    selected = locations[torch.randperm(locations.numel(), device=descriptor.device, generator=generator)[:count]]
    left = descriptor[0].reshape(descriptor.shape[1], -1)[:, selected].T
    right = augmented_descriptor[0].reshape(augmented_descriptor.shape[1], -1)[:, selected].T
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    losses = []
    labels = torch.arange(count, device=descriptor.device)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        logits = left[start:end] @ right.T / temperature
        losses.append(F.cross_entropy(logits, labels[start:end]))
    return torch.stack(losses).mean()


def train_masr(
    manifest: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    resume: Optional[Path] = None,
) -> Dict[str, Any]:
    protocol = str(config.get("run_protocol", "legacy"))
    if protocol not in {"legacy", "controlled_200", "protocol_300"}:
        raise ValueError(f"Unknown MASR run protocol: {protocol!r}")
    frozen_protocol = protocol in {"controlled_200", "protocol_300"}
    if frozen_protocol:
        required_epochs = 200 if protocol == "controlled_200" else 300
        required = {"epochs": required_epochs, "steps_per_epoch": 85, "min_epochs": 120, "patience": 40}
        observed = {key: int(config.get(key, 0)) for key in required}
        if observed != required:
            raise ValueError(f"{protocol} requires {required}, got {observed}.")
    if protocol == "protocol_300" and not config.get("tensorboard_log_dir"):
        raise ValueError("protocol_300 requires an explicit top-level tensorboard_log_dir.")
    tasks = load_registration_manifest(manifest, require_files=True)
    train_splits = {str(v).lower() for v in config.get("train_splits", ["train"])}
    validation_splits = {str(v).lower() for v in config.get("validation_splits", ["validation", "val"])}
    paths = sorted({image.path for task in tasks if task.split in train_splits for image in (task.fixed, task.moving)})
    validation_paths = sorted({image.path for task in tasks if task.split in validation_splits for image in (task.fixed, task.moving)})
    if not paths:
        raise ValueError("No standalone training images selected from the registration manifest.")
    if not validation_paths:
        raise ValueError("MASR training requires standalone label-free validation images for checkpoint selection.")
    overlap = {str(path.resolve()) for path in paths}.intersection(str(path.resolve()) for path in validation_paths)
    if overlap:
        raise ValueError(f"Train/validation image leakage detected: {sorted(overlap)[:3]}")
    if frozen_protocol and (len(paths), len(validation_paths)) != (84, 5):
        raise ValueError(
            f"{protocol} requires 84 train and 5 validation standalone volumes, got "
            f"{len(paths)} and {len(validation_paths)}."
        )
    seed = int(config.get("seed", 2024))
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model = MASRNet(
        channels=config.get("channels", [8, 16, 32, 64]),
        feature_channels=int(config.get("feature_channels", 4)),
        descriptor_channels=int(config.get("descriptor_channels", 24)),
        dns_dilation=int(config.get("dns_dilation", 2)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    start_epoch = 0
    best_loss = float("inf")
    global_step = 0
    bad_epochs = 0
    validation_history = []
    if resume is not None:
        state = torch.load(resume, map_location=device)
        if state.get("schema") != "mok_masr_checkpoint_v1":
            raise ValueError("Resume checkpoint is not a MASR reproduction checkpoint.")
        if state.get("run_protocol", protocol) != protocol:
            raise ValueError("Resume checkpoint run protocol does not match this training run.")
        recorded_manifest = dict(state.get("training_provenance", {})).get("manifest_sha256")
        if recorded_manifest and recorded_manifest != sha256_file(manifest.resolve()):
            raise ValueError("Resume checkpoint was created from a different training manifest.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        recorded_scaler = state.get("scaler_state_dict")
        if isinstance(recorded_scaler, dict) and recorded_scaler:
            scaler.load_state_dict(recorded_scaler)
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_validation_loss", state.get("best_loss", best_loss)))
        global_step = int(state.get("global_step", 0))
        bad_epochs = int(dict(state.get("early_stopping", {})).get("bad_epochs", 0))
        validation_history = [float(value) for value in state.get("validation_history", [])]
        if state.get("python_rng_state") is not None:
            rng.setstate(state["python_rng_state"])
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if state.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda_rng_state"]])
    tensorboard_log_dir = (
        Path(str(config["tensorboard_log_dir"])).resolve()
        if config.get("tensorboard_log_dir")
        else None
    )
    if protocol == "protocol_300" and resume is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"Fresh protocol_300 output directory is not empty: {output_dir.resolve()}")
        if tensorboard_log_dir.exists() and any(tensorboard_log_dir.iterdir()):
            raise ValueError(
                f"Fresh protocol_300 TensorBoard directory is not empty: {tensorboard_log_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "epoch_metrics.jsonl"
    iteration_metrics_path = output_dir / "iteration_metrics.jsonl"
    writer = create_summary_writer(output_dir, log_dir=tensorboard_log_dir)
    epochs = int(config.get("epochs", 100))
    steps = int(config.get("steps_per_epoch", len(paths))) or len(paths)
    crop = tuple(int(v) for v in config.get("crop_shape_zyx", [96, 96, 96]))
    base_lr = float(config.get("learning_rate", 1e-4))
    warmup_steps = int(config.get("warmup_epochs", 10)) * steps
    total_steps = epochs * steps
    min_epochs = int(config.get("min_epochs", 0))
    patience = int(config.get("patience", 0))
    preflight_max_steps = int(config.get("preflight_max_optimizer_steps", 0))
    invocation_start_global_step = global_step
    if frozen_protocol:
        _write_run_lock(output_dir / "RUN_LOCK.json", {
            "schema": "masr_controlled_run_lock_v2", "run_protocol": protocol,
            "method": "masr_dns_io", "epochs": epochs, "batch_size": 1,
            "logical_batches_per_epoch": 85, "unique_train_anchors": 84,
            "rotating_repeat_per_epoch": 1,
            "partner_policy": "self-supervised nonlinear-intensity view of each standalone anchor",
            "updates_per_logical_batch": 1,
            "expected_final_logical_step": total_steps, "expected_final_optimizer_step": total_steps,
            "training_crop_zyx": list(crop), "inference_full_fov": True,
            "optimizer": "Adam", "learning_rate": base_lr, "lr_schedule": "cosine",
            "warmup_epochs": int(config.get("warmup_epochs", 10)), "total_optimizer_steps": total_steps,
            "amp_enabled": False,
            "validation": {"unique_volumes": 5, "labels_used": False, "selection_metric": "label_free_validation_contrastive_loss"},
            "early_stopping": {"applied": False, "reason": f"fixed_{epochs}_budget", "diagnostic_min_epochs": min_epochs, "diagnostic_patience": patience},
            "iteration_logging": {
                "tensorboard": True,
                "tensorboard_log_dir": str(
                    tensorboard_log_dir if tensorboard_log_dir is not None
                    else (output_dir.resolve() / "tf-logs")
                ),
                "jsonl": "iteration_metrics.jsonl",
                "fields": ["contrastive_loss", "learning_rate", "logical_step", "optimizer_step"],
                "images_logged": False,
            },
            "tensorboard_log_dir": str(
                tensorboard_log_dir if tensorboard_log_dir is not None
                else (output_dir.resolve() / "tf-logs")
            ),
            "checkpoint_every_epochs": int(config.get("checkpoint_every_epochs", 50)), "seed": seed,
            "manifest": str(manifest.resolve()), "manifest_sha256": sha256_file(manifest.resolve()),
            "trainer_sha256": sha256_file(Path(__file__).resolve()),
            "implementation_status": "paper_reproduction_no_complete_official_release",
            "literature_budget_deviation": "MASR-Net total training budget is not disclosed; protocol_300 is a controlled rerun, not paper-full.",
            "public8_policy": f"not loaded during training; infer exactly once after epoch {epochs}",
            "preflight_execution_cap": preflight_max_steps,
        })

    def set_lr(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        return float(lr)

    def contrastive_for(image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        augmented = stochastic_nonlinear_transform(
            image, n_control_points=int(config.get("n_control_points", 3)),
            inversion_threshold=float(config.get("inversion_threshold", 0.5)), generator=generator,
        )
        descriptor = model(image); augmented_descriptor = model(augmented)
        return anatomy_contrastive_loss(
            descriptor, augmented_descriptor, image > float(config.get("foreground_threshold", 0.02)),
            n_samples=int(config.get("n_samples", 8196)), temperature=float(config.get("temperature", 0.07)),
            chunk_size=int(config.get("contrastive_chunk_size", 512)), generator=generator,
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        if frozen_protocol:
            epoch_paths = list(paths); rng.shuffle(epoch_paths)
            repeated_path = paths[epoch % len(paths)]
            epoch_paths.append(repeated_path)
        else:
            epoch_paths = []
            while len(epoch_paths) < steps:
                cycle = list(paths); rng.shuffle(cycle); epoch_paths.extend(cycle)
        for path in epoch_paths[:steps]:
            if preflight_max_steps and global_step - invocation_start_global_step >= preflight_max_steps:
                break
            data = robust_unit_interval(load_scalar(path).data_xyz)
            image = _crop_or_pad(data, crop, rng).to(device)
            learning_rate = set_lr(global_step)
            loss = contrastive_for(image)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            detached_loss = float(loss.detach().cpu())
            losses.append(detached_loss)
            iteration_record = {
                "epoch": epoch,
                "contrastive_loss": detached_loss,
                "learning_rate": learning_rate,
                "logical_step": global_step,
                "optimizer_step": global_step,
            }
            with iteration_metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(iteration_record, ensure_ascii=False) + "\n")
            log_iteration(writer, iteration_record, global_step)
        epoch_loss = float(np.mean(losses))
        model.eval(); validation_losses = []
        with torch.no_grad():
            for index, path in enumerate(validation_paths):
                deterministic_rng = random.Random(seed + 100000 + index)
                data = robust_unit_interval(load_scalar(path).data_xyz)
                image = _crop_or_pad(data, crop, deterministic_rng).to(device)
                generator = torch.Generator(device=device).manual_seed(seed + 200000 + index)
                validation_losses.append(float(contrastive_for(image, generator).detach().cpu()))
        validation_loss = float(np.mean(validation_losses))
        validation_history.append(validation_loss)
        convergence = _convergence_diagnostic(validation_history)
        improved = validation_loss < best_loss
        bad_epochs = 0 if improved else bad_epochs + 1
        would_stop_early = bool(patience > 0 and epoch + 1 >= min_epochs and bad_epochs >= patience)
        record = {
            "epoch": epoch, "train_contrastive_loss": epoch_loss,
            "validation_contrastive_loss": validation_loss, "steps": steps,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "logical_step": global_step, "optimizer_step": global_step,
            "early_stop_bad_epochs": bad_epochs, "would_stop_early": would_stop_early,
            "unique_anchors": len(paths), "logical_batches_configured": steps,
            "logical_batches_executed": len(losses),
            "rotating_repeat": 1 if frozen_protocol else 0,
            "rotating_repeat_anchor": str(repeated_path.resolve()) if frozen_protocol else None,
        }
        if convergence["available"]:
            record.update({
                "last20_validation_relative_slope": convergence["relative_slope_per_epoch"],
                "last20_not_converged": convergence["not_converged_still_improving"],
            })
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_record(writer, record, epoch)
        state = {
            "schema": "mok_masr_checkpoint_v1",
            "epoch": epoch,
            "global_step": global_step,
            "logical_step": global_step,
            "optimizer_step": global_step,
            "best_validation_loss": min(best_loss, validation_loss),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": {
                "implementation": "manual", "name": "cosine", "last_step": global_step,
                "total_steps": total_steps, "warmup_steps": warmup_steps, "base_lr": base_lr,
            },
            "scaler_state_dict": scaler.state_dict(), "amp_enabled": False,
            "architecture": model.architecture,
            "training_config": dict(config),
            "run_config": dict(config),
            "run_protocol": protocol,
            "early_stopping": {
                "min_epochs": min_epochs, "patience": patience,
                "bad_epochs": bad_epochs, "would_trigger": would_stop_early,
                "applied": False, "budget_policy": f"fixed_{epochs}_budget",
            },
            "validation_history": validation_history,
            "convergence_diagnostic": convergence,
            "sampling_state": {
                "unique_anchors": len(paths), "logical_batches": steps,
                "rotating_repeat": 1 if frozen_protocol else 0,
                "rotating_repeat_anchor": str(repeated_path.resolve()) if frozen_protocol else None,
            },
            "paper_disclosed": {"n_samples": 8196, "temperature": 0.07, "n_control_points": 3, "inversion_threshold": 0.5, "descriptor_channels": 24},
            "reproduction_assumptions": {"dns_dilation_voxels": int(config.get("dns_dilation", 2))},
            "selection_metric": "label_free_validation_contrastive_loss",
            "python_rng_state": rng.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_provenance": {
                "manifest": str(manifest.resolve()), "manifest_sha256": sha256_file(manifest.resolve()),
                "source_revision": "local_paper_reproduction",
                "trainer_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        save_training_checkpoints(
            state,
            output_dir,
            epoch=epoch,
            improved=improved,
            checkpoint_every=int(config.get("checkpoint_every_epochs", 50)),
        )
        if improved:
            best_loss = validation_loss
        if preflight_max_steps and global_step - invocation_start_global_step >= preflight_max_steps:
            break
    if frozen_protocol and not preflight_max_steps and global_step != total_steps:
        raise RuntimeError(
            f"{protocol} ended with incorrect step count: {global_step} != {total_steps}."
        )
    writer.close()
    return {"epochs": epochs, "best_validation_loss": best_loss, "last_checkpoint": str(output_dir / "last.pt"), "best_checkpoint": str(output_dir / "best.pt")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("DNS training config must be a YAML mapping.")
    print(json.dumps(train_masr(args.manifest, args.output_dir, config, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
