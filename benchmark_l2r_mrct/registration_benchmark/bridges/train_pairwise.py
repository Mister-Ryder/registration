"""Unified label-free trainer for DGMIR-U and MIND capacity controls.

This does not replace the upstream repositories. It imports their released
architectures and changes only data access/checkpointing so labels never enter
training. DGMIR-U explicitly removes the released Dice term.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..contract import canonical_json_hash, load_registration_manifest
from .common import capture_rng_state, gradient_loss, load_pair_tensors, restore_rng_state, warp
from ..provenance import git_identity, sha256_file
from ..tensorboard_logging import create_summary_writer, log_iteration, log_record
from ..checkpointing import save_training_checkpoints


def _load_architecture(method: str, repo: Path, shape, device):
    if method == "dgmir_u":
        sys.path.insert(0, str(repo))
        from model2 import DGMIR
        return DGMIR(list(shape)).to(device), None
    if method == "transmorph_mind":
        source = repo / "TransMorph"
        sys.path.insert(0, str(source))
        from models.TransMorph import CONFIGS, TransMorph
        config = CONFIGS["TransMorph"]
        config.img_size = tuple(shape)
        return TransMorph(config).to(device), None
    source = repo / "CorrMLP"
    sys.path.insert(0, str(source))
    from networks import CorrMLP
    return CorrMLP(use_checkpoint=True).to(device), None


def _mind_loss(repo_dgmir: Path, device):
    sys.path.insert(0, str(repo_dgmir))
    from losses import MINDSSCLoss
    return MINDSSCLoss(radius=2, dilation=2, penalty="l2").to(device)


def _forward(method, model, wrapper, fixed, moving, model_shape_zyx):
    if method == "dgmir_u":
        flow, warped = model(fixed, moving, "train" if model.training else "test")
        return flow, warped
    if method == "transmorph_mind":
        warped, flow = model(torch.cat([moving, fixed], dim=1))
        return flow, warped
    if method == "corrmlp_mind":
        warped, flow = model(fixed, moving)
        return flow, warped
    wrapper.identity_map.isIdentity = True
    mapping = model(moving, fixed)
    coordinates = mapping(wrapper.identity_map)
    factors = torch.as_tensor(
        [max(v - 1, 1) for v in model_shape_zyx], device=coordinates.device, dtype=coordinates.dtype
    )
    flow = (coordinates - wrapper.identity_map) * factors[None, :, None, None, None]
    return flow, warp(moving, flow)


def _label_free_loss(method, model, wrapper, mind, task, device, model_shape_zyx, smoothness_weight):
    _, _, fixed, moving = load_pair_tensors(task.fixed.path, task.moving.path, device, model_shape_zyx)
    flow, warped = _forward(method, model, wrapper, fixed, moving, model_shape_zyx)
    similarity = mind(fixed, warped)
    smoothness = gradient_loss(flow)
    return similarity + smoothness_weight * smoothness, similarity, smoothness


def _learning_rate(step, total_steps, base_lr, warmup_steps, schedule):
    if warmup_steps and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "poly":
        return base_lr * (1.0 - progress) ** 0.9
    return base_lr


def _serializable_arguments(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        # Resume is invocation state, not part of the frozen experiment.  If it
        # enters RUN_LOCK, a legitimate restart can never match the first call.
        if key != "resume"
    }


def _unique_split_images(tasks):
    images = {}
    for task in tasks:
        for image in (task.fixed, task.moving):
            images[str(image.path.resolve())] = image
    return [images[key] for key in sorted(images)]


def _controlled_epoch_pairs(tasks, steps: int, epoch: int):
    """Build 84 unique cross-modal anchors plus one deterministic rotation.

    Every train volume is a fixed/source anchor exactly once.  The 85th anchor
    rotates over the canonical 84-volume list by epoch, while each partner is
    sampled from the opposite modality using the checkpointed Python RNG.
    """

    canonical_anchors = _unique_split_images(tasks)
    if steps != 85 or len(canonical_anchors) != 84:
        raise ValueError(
            "The frozen capacity protocol requires 84 unique train anchors and 85 logical "
            f"batches, got {len(canonical_anchors)} anchors and {steps} batches."
        )
    partners_by_modality = {
        modality: [image for image in canonical_anchors if image.modality == modality]
        for modality in ("mr", "ct")
    }
    if [len(partners_by_modality[key]) for key in ("mr", "ct")] != [38, 46]:
        raise ValueError(
            "The frozen capacity protocol expects the train split to contain 38 MR "
            f"and 46 CT volumes, got {[len(partners_by_modality[key]) for key in ('mr', 'ct')]}."
        )
    epoch_anchors = list(canonical_anchors)
    random.shuffle(epoch_anchors)
    repeated_anchor = canonical_anchors[epoch % len(canonical_anchors)]
    epoch_anchors.append(repeated_anchor)
    template = tasks[0]
    scheduled = []
    schedule_rows = []
    for logical_index, anchor in enumerate(epoch_anchors):
        opposite = "ct" if anchor.modality == "mr" else "mr"
        partner = random.choice(partners_by_modality[opposite])
        scheduled.append(replace(
            template,
            pair_id=f"controlled_e{epoch:03d}_l{logical_index:03d}",
            subject_id=f"{anchor.path.name}__{partner.path.name}",
            fixed=anchor,
            moving=partner,
        ))
        schedule_rows.append({
            "logical_index": logical_index,
            "anchor": str(anchor.path.resolve()),
            "anchor_modality": anchor.modality,
            "partner": str(partner.path.resolve()),
            "partner_modality": partner.modality,
        })
    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return scheduled, {
        "unique_anchors": 84,
        "logical_batches": 85,
        "rotating_repeat": 1,
        "rotating_repeat_anchor": str(repeated_anchor.path.resolve()),
        "mr_anchors": 38,
        "ct_anchors": 46,
        "schedule_sha256": schedule_sha256,
    }


def _write_run_lock(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["content_sha256"] = canonical_json_hash(payload)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"RUN_LOCK mismatch: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def _convergence_diagnostic(history) -> dict:
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["dgmir_u", "transmorph_mind", "corrmlp_mind"])
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dgmir-repo", required=True, type=Path, help="Pinned source of the shared MIND-SSC implementation.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr-schedule", choices=["cosine", "poly", "constant"], default="cosine")
    parser.add_argument("--smoothness-weight", type=float, default=0.5)
    parser.add_argument("--model-shape-zyx", nargs=3, type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument(
        "--run-protocol",
        choices=["legacy", "controlled_200", "protocol_300"],
        default="legacy",
    )
    parser.add_argument("--preflight-max-optimizer-steps", type=int, default=0)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.run_protocol == "controlled_200":
        required = {"epochs": 200, "steps_per_epoch": 85, "min_epochs": 120, "patience": 40}
        observed = {key: getattr(args, key) for key in required}
        if observed != required:
            raise ValueError(f"controlled_200 requires {required}, got {observed}.")
    if args.run_protocol == "protocol_300":
        required = {"epochs": 300, "steps_per_epoch": 85, "min_epochs": 120, "patience": 40}
        observed = {key: getattr(args, key) for key in required}
        if observed != required:
            raise ValueError(f"protocol_300 requires {required}, got {observed}.")
        if args.method not in {"dgmir_u", "transmorph_mind", "corrmlp_mind"}:
            raise ValueError("protocol_300 is frozen for B08/B11/B12.")
        if args.tensorboard_dir is None:
            raise ValueError("protocol_300 requires an explicit top-level --tensorboard-dir.")
    capacity_protocol = args.run_protocol in {"controlled_200", "protocol_300"}
    manifest_tasks = load_registration_manifest(args.manifest)
    tasks = [task for task in manifest_tasks if task.split == "train"]
    validation_tasks = [task for task in manifest_tasks if task.split in {"validation", "val"}]
    if not tasks:
        raise ValueError("No train pairs in manifest.")
    if not validation_tasks:
        raise ValueError("A label-free validation split is required for checkpoint selection.")
    train_image_paths = {str(image.path.resolve()) for task in tasks for image in (task.fixed, task.moving)}
    validation_image_paths = {
        str(image.path.resolve()) for task in validation_tasks for image in (task.fixed, task.moving)
    }
    overlap = train_image_paths.intersection(validation_image_paths)
    if overlap:
        raise ValueError(f"Train/validation image leakage detected: {sorted(overlap)[:3]}")
    if capacity_protocol and (len(train_image_paths), len(validation_image_paths)) != (84, 5):
        raise ValueError(
            f"{args.run_protocol} requires 84 train and 5 validation volumes, got "
            f"{len(train_image_paths)} and {len(validation_image_paths)}."
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.method == "dgmir_u" and device.type != "cuda":
        raise RuntimeError("Released DGMIR requires CUDA.")
    model, wrapper = _load_architecture(args.method, args.repo.resolve(), tuple(args.model_shape_zyx), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mind = _mind_loss(args.dgmir_repo.resolve(), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    updates_per_logical = 2 if capacity_protocol and args.method == "transmorph_mind" else 1
    if args.preflight_max_optimizer_steps and args.preflight_max_optimizer_steps % updates_per_logical:
        raise ValueError("Preflight optimizer-step cap must end on a logical-batch boundary.")
    start_epoch, best = 0, float("inf")
    logical_step, optimizer_step, bad_epochs, validation_history = 0, 0, 0, []
    scheduler_state = {}
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if state.get("method") != args.method or list(state.get("model_shape_zyx", [])) != list(args.model_shape_zyx):
            raise ValueError("Resume checkpoint method/model shape does not match this training run.")
        if state.get("run_protocol", args.run_protocol) != args.run_protocol:
            raise ValueError("Resume checkpoint run protocol does not match this training run.")
        recorded_manifest = dict(state.get("training_provenance", {})).get("manifest_sha256")
        if recorded_manifest and recorded_manifest != sha256_file(args.manifest.resolve()):
            raise ValueError("Resume checkpoint was created from a different training manifest.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        recorded_scaler = state.get("scaler_state_dict")
        if isinstance(recorded_scaler, dict) and recorded_scaler:
            scaler.load_state_dict(recorded_scaler)
        start_epoch, best = int(state["epoch"]) + 1, float(state["best_validation_loss"])
        logical_step = int(state.get("logical_step", state.get("global_step", 0)))
        optimizer_step = int(state.get("optimizer_step", state.get("global_step", logical_step)))
        scheduler_state = dict(state.get("scheduler_state_dict", {}))
        if scheduler_state and int(scheduler_state.get("updates_per_logical", updates_per_logical)) != updates_per_logical:
            raise ValueError("Resume checkpoint update-count schedule does not match this run.")
        bad_epochs = int(dict(state.get("early_stopping", {})).get("bad_epochs", 0))
        validation_history = [float(value) for value in state.get("validation_history", [])]
        restore_rng_state(state.get("rng_state"))
    # The preflight cap is an invocation-local execution budget.  Treating it as
    # an absolute global optimizer step makes a resumed one-step gate perform no
    # update once the checkpoint counter has already reached the cap.
    invocation_start_optimizer_step = optimizer_step
    if args.run_protocol == "protocol_300" and not args.resume:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(
                f"Fresh protocol_300 output directory is not empty: {args.output_dir.resolve()}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = args.output_dir / "epoch_metrics.jsonl"
    iteration_metrics = args.output_dir / "iteration_metrics.jsonl"
    if args.run_protocol == "protocol_300" and not args.resume:
        if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
            raise ValueError(
                f"Fresh protocol_300 TensorBoard directory is not empty: {args.tensorboard_dir.resolve()}"
            )
    writer = create_summary_writer(args.output_dir, log_dir=args.tensorboard_dir)
    steps = args.steps_per_epoch or len(tasks)
    total_optimizer_steps = args.epochs * steps * updates_per_logical
    warmup_optimizer_steps = args.warmup_epochs * steps * updates_per_logical
    if scheduler_state:
        expected_scheduler = {
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_optimizer_steps": warmup_optimizer_steps,
            "updates_per_logical": updates_per_logical,
        }
        observed_scheduler = {key: int(scheduler_state.get(key, -1)) for key in expected_scheduler}
        if observed_scheduler != expected_scheduler:
            raise ValueError(
                f"Resume checkpoint scheduler mismatch: {observed_scheduler} != {expected_scheduler}."
            )
    architecture_identity = git_identity(args.repo.resolve())
    mind_identity = git_identity(args.dgmir_repo.resolve())
    if capacity_protocol:
        if architecture_identity["dirty"] or mind_identity["dirty"]:
            raise RuntimeError(f"{args.run_protocol} requires clean frozen official source worktrees.")
        architecture_names = {
            "dgmir_u": "official DGMIR architecture with label-free unsupervised fairness adaptation",
            "transmorph_mind": "official TransMorph architecture with MIND-SSC",
            "corrmlp_mind": "official CorrMLP architecture with MIND-SSC",
        }
        if args.run_protocol == "controlled_200":
            literature_deviations = {
                "dgmir_u": "DGMIR-U removes released label/Dice supervision and uses controlled_200; it is an adapted fair rerun, not paper-full.",
                "transmorph_mind": "TransMorph official training uses 500 epochs; controlled_200 is not paper-full.",
                "corrmlp_mind": "CorrMLP literature/release uses 100k/40k-scale settings; 17k updates are not paper-full.",
            }
        else:
            literature_deviations = {
                "dgmir_u": "DGMIR-U removes released label/Dice supervision and uses protocol_300; it is an adapted fair rerun, not paper-full.",
                "transmorph_mind": "TransMorph official training uses 500 epochs; protocol_300 is not paper-full.",
                "corrmlp_mind": "CorrMLP literature/release uses 100k/40k-scale settings; 25.5k updates are not paper-full.",
            }
        expected_logical_step = args.epochs * steps
        fixed_budget_reason = f"fixed_{args.epochs}_budget"
        _write_run_lock(args.output_dir / "RUN_LOCK.json", {
            "schema": "capacity_control_run_lock_v2",
            "run_protocol": args.run_protocol,
            "method": args.method,
            "architecture": architecture_names[args.method],
            "epochs": args.epochs,
            "batch_size": 1,
            "logical_batches_per_epoch": 85,
            "unique_train_anchors": 84,
            "rotating_repeat_per_epoch": 1,
            "partner_policy": "checkpointed-RNG random opposite-modality partner per logical batch",
            "updates_per_logical_batch": updates_per_logical,
            "expected_final_logical_step": expected_logical_step,
            "expected_final_optimizer_step": expected_logical_step * updates_per_logical,
            "model_shape_zyx": list(args.model_shape_zyx),
            "whole_volume_full_fov": True,
            "resolution_policy": {
                "prepared_grid_shape_xyz": [192, 160, 192],
                "prepared_spacing_mm_xyz": [2.0, 2.0, 2.0],
                "model_input_shape_zyx": list(args.model_shape_zyx),
                "training_patch": "whole volume",
                "flow_resize_factor_dzyx": [1.0, 1.0, 1.0],
            },
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "lr_schedule": args.lr_schedule,
            "warmup_epochs": args.warmup_epochs,
            "total_optimizer_steps": total_optimizer_steps,
            "smoothness_weight": args.smoothness_weight,
            "amp_enabled": False,
            "validation": {
                "unique_volumes": 5,
                "pair_calls": len(validation_tasks),
                "selection_metric": "label_free_validation_loss",
                "labels_used": False,
            },
            "early_stopping": {
                "applied": False,
                "reason": fixed_budget_reason,
                "diagnostic_min_epochs": args.min_epochs,
                "diagnostic_patience": args.patience,
            },
            "iteration_logging": {
                "tensorboard": True,
                "tensorboard_log_dir": str(
                    args.tensorboard_dir.resolve()
                    if args.tensorboard_dir is not None
                    else (args.output_dir.resolve() / "tf-logs")
                ),
                "jsonl": "iteration_metrics.jsonl",
                "fields": [
                    "loss", "mind_loss", "smoothness_loss", "learning_rate",
                    "logical_step", "optimizer_step", "direction",
                ],
                "images_logged": False,
            },
            "tensorboard_log_dir": str(
                args.tensorboard_dir.resolve()
                if args.tensorboard_dir is not None
                else (args.output_dir.resolve() / "tf-logs")
            ),
            "checkpoint_every_epochs": args.checkpoint_every,
            "seed": args.seed,
            "run_config": _serializable_arguments(args),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest.resolve()),
            "architecture_repo": architecture_identity,
            "mind_repo": mind_identity,
            "trainer_source": str(Path(__file__).resolve()),
            "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
            "literature_budget_deviation": literature_deviations[args.method],
            "public8_policy": f"not loaded during training; infer exactly once after epoch {args.epochs}",
            "preflight_execution_cap": args.preflight_max_optimizer_steps,
        })
    for epoch in range(start_epoch, args.epochs):
        model.train(); values = []; direction_values = {"mr_from_ct": [], "ct_from_mr": []}
        if capacity_protocol:
            epoch_tasks, sampling = _controlled_epoch_pairs(tasks, steps, epoch)
        else:
            epoch_tasks = sorted(tasks, key=lambda value: value.pair_id)
            random.shuffle(epoch_tasks)
            sampling = {
                "unique_anchors": len({task.fixed.path for task in epoch_tasks}),
                "logical_batches": steps,
                "rotating_repeat": 0,
            }
        for step in range(steps):
            if (
                args.preflight_max_optimizer_steps
                and optimizer_step - invocation_start_optimizer_step >= args.preflight_max_optimizer_steps
            ):
                break
            task = epoch_tasks[step % len(epoch_tasks)]
            update_tasks = [task]
            if capacity_protocol and args.method == "transmorph_mind":
                update_tasks.append(replace(
                    task,
                    pair_id=f"{task.pair_id}__reverse",
                    fixed=task.moving,
                    moving=task.fixed,
                ))
            for update_index, update_task in enumerate(update_tasks, start=1):
                lr = _learning_rate(
                    optimizer_step, total_optimizer_steps, args.learning_rate,
                    warmup_optimizer_steps, args.lr_schedule,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss, similarity, smoothness = _label_free_loss(
                    args.method, model, wrapper, mind, update_task, device,
                    args.model_shape_zyx, args.smoothness_weight,
                )
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                detached = (float(loss.detach()), float(similarity.detach()), float(smoothness.detach()))
                values.append(detached)
                direction = (
                    "mr_from_ct"
                    if update_task.fixed.modality == "mr" and update_task.moving.modality == "ct"
                    else "ct_from_mr"
                )
                direction_values[direction].append(detached[0])
                optimizer_step += 1
                iteration_record = {
                    "epoch": epoch,
                    "logical_step": logical_step + 1,
                    "optimizer_step": optimizer_step,
                    "update_index_in_logical_batch": update_index,
                    "updates_per_logical_batch": updates_per_logical,
                    "direction": direction,
                    "loss": detached[0],
                    "mind_loss": detached[1],
                    "smoothness_loss": detached[2],
                    "learning_rate": float(lr),
                    "segmentation_labels_used": False,
                }
                with iteration_metrics.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(iteration_record) + "\n")
                log_iteration(
                    writer,
                    {
                        "loss": iteration_record["loss"],
                        "mind_loss": iteration_record["mind_loss"],
                        "smoothness_loss": iteration_record["smoothness_loss"],
                        "learning_rate": iteration_record["learning_rate"],
                        "logical_step": iteration_record["logical_step"],
                        "optimizer_step": iteration_record["optimizer_step"],
                        "direction_mr_from_ct": float(direction == "mr_from_ct"),
                    },
                    optimizer_step,
                )
            logical_step += 1
        train_loss = float(np.mean([v[0] for v in values]))
        model.eval(); validation_values = []
        with torch.no_grad():
            for task in validation_tasks:
                value, _, _ = _label_free_loss(
                    args.method, model, wrapper, mind, task, device, args.model_shape_zyx, args.smoothness_weight
                )
                validation_values.append(float(value.detach()))
        validation_loss = float(np.mean(validation_values))
        validation_history.append(validation_loss)
        convergence = _convergence_diagnostic(validation_history)
        improved = validation_loss < best
        bad_epochs = 0 if improved else bad_epochs + 1
        would_stop_early = bool(
            args.patience > 0 and epoch + 1 >= args.min_epochs and bad_epochs >= args.patience
        )
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "validation_loss": validation_loss,
            "mind_loss": float(np.mean([v[1] for v in values])),
            "smoothness_loss": float(np.mean([v[2] for v in values])),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "segmentation_labels_used": False,
            "logical_step": logical_step, "optimizer_step": optimizer_step,
            "updates_per_logical_batch": updates_per_logical,
            "unique_anchors": sampling["unique_anchors"],
            "logical_batches_configured": sampling["logical_batches"],
            "logical_batches_executed": len(values) // updates_per_logical,
            "rotating_repeat": sampling["rotating_repeat"],
            "sampling_schedule_sha256": sampling.get("schedule_sha256"),
            "mr_from_ct_updates": len(direction_values["mr_from_ct"]),
            "ct_from_mr_updates": len(direction_values["ct_from_mr"]),
            "mr_from_ct_loss": float(np.mean(direction_values["mr_from_ct"])) if direction_values["mr_from_ct"] else None,
            "ct_from_mr_loss": float(np.mean(direction_values["ct_from_mr"])) if direction_values["ct_from_mr"] else None,
            "early_stop_bad_epochs": bad_epochs, "would_stop_early": would_stop_early,
        }
        if convergence["available"]:
            record.update({
                "last20_validation_relative_slope": convergence["relative_slope_per_epoch"],
                "last20_not_converged": convergence["not_converged_still_improving"],
            })
        if device.type == "cuda":
            record.update({
                "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
                "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
            })
        with metrics.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        log_record(writer, record, epoch)
        state = {
            "schema": "comparison_pairwise_checkpoint_v1", "method": args.method,
            "epoch": epoch, "global_step": optimizer_step,
            "logical_step": logical_step, "optimizer_step": optimizer_step,
            "best_validation_loss": min(best, validation_loss),
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": {
                "implementation": "manual", "name": args.lr_schedule, "last_optimizer_step": optimizer_step,
                "total_optimizer_steps": total_optimizer_steps,
                "warmup_optimizer_steps": warmup_optimizer_steps,
                "updates_per_logical": updates_per_logical,
                "base_lr": args.learning_rate,
            },
            "scaler_state_dict": scaler.state_dict(), "amp_enabled": False,
            "model_shape_zyx": list(args.model_shape_zyx), "segmentation_labels_used": False,
            "loss": {"similarity": "MIND-SSC radius=2 dilation=2", "smoothness_weight": args.smoothness_weight},
            "selection_metric": "label_free_validation_loss",
            "schedule": {"name": args.lr_schedule, "warmup_epochs": args.warmup_epochs, "base_lr": args.learning_rate},
            "run_protocol": args.run_protocol,
            "run_config": _serializable_arguments(args),
            "early_stopping": {
                "min_epochs": args.min_epochs, "patience": args.patience,
                "bad_epochs": bad_epochs, "would_trigger": would_stop_early,
                "applied": False, "reason": f"fixed_{args.epochs}_budget",
            },
            "sampling_state": sampling,
            "validation_history": validation_history,
            "convergence_diagnostic": convergence,
            "rng_state": capture_rng_state(),
            "training_provenance": {
                "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest.resolve()),
                "architecture_repo": architecture_identity,
                "mind_repo": mind_identity,
            },
        }
        save_training_checkpoints(
            state, args.output_dir, epoch=epoch, improved=improved,
            checkpoint_every=args.checkpoint_every,
        )
        if improved:
            best = validation_loss
        if (
            args.preflight_max_optimizer_steps
            and optimizer_step - invocation_start_optimizer_step >= args.preflight_max_optimizer_steps
        ):
            break
    if capacity_protocol and not args.preflight_max_optimizer_steps:
        expected_logical = args.epochs * steps
        expected_optimizer = expected_logical * updates_per_logical
        if (logical_step, optimizer_step) != (expected_logical, expected_optimizer):
            raise RuntimeError(
                f"{args.run_protocol} ended with incorrect step counts: "
                f"{(logical_step, optimizer_step)} != {(expected_logical, expected_optimizer)}."
            )
    writer.close()
    sys.path = [entry for entry in sys.path if entry not in {str(args.repo.resolve()), str(args.dgmir_repo.resolve())}]


if __name__ == "__main__":
    main()
