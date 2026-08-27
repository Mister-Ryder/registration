"""Label-free M2M-Reg training on benchmark manifests using official M2M code."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..contract import load_registration_manifest
from .common import capture_rng_state, load_pair_tensors, restore_rng_state
from .train_pairwise import (
    _controlled_epoch_pairs, _convergence_diagnostic, _learning_rate, _serializable_arguments,
    _unique_split_images, _write_run_lock,
)
from ..provenance import git_identity, sha256_file
from ..tensorboard_logging import create_summary_writer, log_iteration, log_record
from ..checkpointing import save_training_checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-shape-zyx", required=True, nargs=3, type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr-schedule", choices=["cosine", "poly", "constant"], default="cosine")
    parser.add_argument("--lambda-inverse", type=float, default=0.5)
    parser.add_argument("--lambda-canonical", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--run-protocol", choices=["legacy", "controlled_200", "protocol_300"], default="legacy")
    parser.add_argument("--preflight-max-optimizer-steps", type=int, default=0)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    frozen_protocol = args.run_protocol in {"controlled_200", "protocol_300"}
    if frozen_protocol:
        required_epochs = 200 if args.run_protocol == "controlled_200" else 300
        required = {"epochs": required_epochs, "steps_per_epoch": 85, "min_epochs": 120, "patience": 40}
        observed = {key: getattr(args, key) for key in required}
        if observed != required:
            raise ValueError(f"{args.run_protocol} requires {required}, got {observed}.")
    if args.run_protocol == "protocol_300" and args.tensorboard_dir is None:
        raise ValueError("protocol_300 requires an explicit top-level --tensorboard-dir.")
    tasks = load_registration_manifest(args.manifest)
    training = [task for task in tasks if task.split == "train"]
    validation = [task for task in tasks if task.split in {"validation", "val"}]
    if len(training) < 3 or not validation:
        raise ValueError("M2M-Reg requires at least three training pairs and a label-free validation split.")
    train_images = _unique_split_images(training); validation_images = _unique_split_images(validation)
    train_paths = {str(image.path.resolve()) for image in train_images}
    validation_paths = {str(image.path.resolve()) for image in validation_images}
    overlap = train_paths.intersection(validation_paths)
    if overlap:
        raise ValueError(f"Train/validation image leakage detected: {sorted(overlap)[:3]}")
    if frozen_protocol and (len(train_images), len(validation_images)) != (84, 5):
        raise ValueError(
            f"{args.run_protocol} requires 84 train and 5 validation volumes, got "
            f"{len(train_images)} and {len(validation_images)}."
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The released M2M/ICON stack is prepared and supported in its CUDA environment.")
    repo = args.repo.resolve(); sys.path.insert(0, str(repo))
    try:
        from models import make_network, make_sim
        from icon_registration import config as icon_config
        icon_config.device = device
        namespace = SimpleNamespace(
            model="gradicon", input_shape=(1, 1, *args.model_shape_zyx), num_cano="-1",
            lambda_inv=args.lambda_inverse, lambda_can=args.lambda_canonical,
            log_mono=False, small=False,
        )
        net = make_network(namespace, include_last_step=False, loss_fn=make_sim("mind"), use_label=False).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        start_epoch, best, bad_epochs, validation_history = 0, float("inf"), 0, []
        logical_step, optimizer_step = 0, 0
        if args.resume:
            state = torch.load(args.resume, map_location=device)
            if state.get("method") != "m2m_reg" or list(state.get("model_shape_zyx", [])) != list(args.model_shape_zyx):
                raise ValueError("Resume checkpoint method/model shape mismatch.")
            if state.get("run_protocol", args.run_protocol) != args.run_protocol:
                raise ValueError("Resume checkpoint run protocol does not match this training run.")
            recorded_manifest = dict(state.get("training_provenance", {})).get("manifest_sha256")
            if recorded_manifest and recorded_manifest != sha256_file(args.manifest.resolve()):
                raise ValueError("Resume checkpoint was created from a different training manifest.")
            net.regis_net.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            recorded_scaler = state.get("scaler_state_dict")
            if isinstance(recorded_scaler, dict) and recorded_scaler:
                scaler.load_state_dict(recorded_scaler)
            start_epoch = int(state["epoch"]) + 1
            logical_step = int(state.get("logical_step", state.get("global_step", 0)))
            optimizer_step = int(state.get("optimizer_step", state.get("global_step", logical_step)))
            best = float(state["best_validation_loss"])
            bad_epochs = int(dict(state.get("early_stopping", {})).get("bad_epochs", 0))
            validation_history = [float(value) for value in state.get("validation_history", [])]
            restore_rng_state(state.get("rng_state"))
        invocation_start_optimizer_step = optimizer_step
        if args.run_protocol == "protocol_300" and not args.resume:
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError(f"Fresh protocol_300 output directory is not empty: {args.output_dir.resolve()}")
            if args.tensorboard_dir.exists() and any(args.tensorboard_dir.iterdir()):
                raise ValueError(f"Fresh protocol_300 TensorBoard directory is not empty: {args.tensorboard_dir.resolve()}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / "epoch_metrics.jsonl"
        iteration_log_path = args.output_dir / "iteration_metrics.jsonl"
        writer = create_summary_writer(args.output_dir, log_dir=args.tensorboard_dir)
        steps = args.steps_per_epoch or len(training)
        total_steps, warmup_steps = args.epochs * steps, args.warmup_epochs * steps
        repo_identity = git_identity(repo)
        if frozen_protocol:
            if repo_identity["dirty"]:
                raise RuntimeError(f"{args.run_protocol} requires a clean frozen M2M-Reg source worktree.")
            _write_run_lock(args.output_dir / "RUN_LOCK.json", {
                "schema": "m2m_controlled_run_lock_v2",
                "run_protocol": args.run_protocol, "method": "m2m_reg",
                "epochs": args.epochs, "batch_size": 1, "logical_batches_per_epoch": 85,
                "unique_train_anchors": 84, "rotating_repeat_per_epoch": 1,
                "partner_policy": "checkpointed-RNG random opposite-modality partner per logical batch",
                "updates_per_logical_batch": 1,
                "expected_final_logical_step": total_steps, "expected_final_optimizer_step": total_steps,
                "model_shape_zyx": list(args.model_shape_zyx), "whole_volume_full_fov": True,
                "resolution_policy": {
                    "prepared_grid_shape_xyz": [192, 160, 192],
                    "prepared_spacing_mm_xyz": [2.0, 2.0, 2.0],
                    "model_input_shape_zyx": list(args.model_shape_zyx),
                    "training_patch": "whole volume",
                },
                "optimizer": "Adam", "learning_rate": args.learning_rate,
                "lr_schedule": args.lr_schedule, "warmup_epochs": args.warmup_epochs,
                "total_optimizer_steps": total_steps,
                "loss": {"similarity": "MIND-SSC", "lambda_inverse": args.lambda_inverse, "lambda_canonical": args.lambda_canonical},
                "amp_enabled": False,
                "validation": {"unique_volumes": 5, "pair_calls": len(validation), "labels_used": False, "selection_metric": "label_free_validation_all_loss"},
                "early_stopping": {"applied": False, "reason": f"fixed_{args.epochs}_budget", "diagnostic_min_epochs": args.min_epochs, "diagnostic_patience": args.patience},
                "iteration_logging": {
                    "tensorboard": True,
                    "tensorboard_log_dir": str(args.tensorboard_dir.resolve() if args.tensorboard_dir is not None else args.output_dir.resolve() / "tf-logs"),
                    "jsonl": "iteration_metrics.jsonl",
                    "fields": ["loss", "mind_similarity", "inverse_consistency", "canonical_cycle_consistency", "learning_rate", "logical_step", "optimizer_step"],
                    "images_logged": False,
                },
                "tensorboard_log_dir": str(args.tensorboard_dir.resolve() if args.tensorboard_dir is not None else args.output_dir.resolve() / "tf-logs"),
                "checkpoint_every_epochs": args.checkpoint_every, "seed": args.seed,
                "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest.resolve()),
                "repo": repo_identity,
                "literature_budget_deviation": "M2M-Reg reports 50,000 iterations; protocol_300 uses 25,500 controlled logical/optimizer steps and is not paper-full.",
                "public8_policy": f"not loaded during training; infer exactly once after epoch {args.epochs}",
                "preflight_execution_cap": args.preflight_max_optimizer_steps,
            })

        def image_pair(task):
            _, _, fixed, moving = load_pair_tensors(task.fixed.path, task.moving.path, device, args.model_shape_zyx)
            return moving, fixed

        def objective(task, canonical_source_task, canonical_target_task):
            moving, fixed = image_pair(task)
            moving_canonical, _ = image_pair(canonical_source_task)
            _, fixed_canonical = image_pair(canonical_target_task)
            # The released wrapper computes Dice only for logging.  A neutral
            # all-one tensor prevents empty-class NaNs; it never enters all_loss.
            neutral = torch.ones_like(moving)
            result = net(moving, fixed, moving_canonical, fixed_canonical, neutral, neutral)
            return result.all_loss.mean(), result.similarity_loss.mean(), result.inverse_consistency_loss.mean(), result.canonical_consistency_loss.mean()

        for epoch in range(start_epoch, args.epochs):
            net.train()
            if frozen_protocol:
                epoch_training, sampling = _controlled_epoch_pairs(training, steps, epoch)
            else:
                epoch_training = sorted(training, key=lambda value: value.pair_id)
                random.shuffle(epoch_training)
                sampling = {"unique_anchors": len(_unique_split_images(training)), "logical_batches": steps, "rotating_repeat": 0}
            values = []
            direction_counts = {"mr_from_ct": 0, "ct_from_mr": 0}
            for index in range(steps):
                if (
                    args.preflight_max_optimizer_steps
                    and optimizer_step - invocation_start_optimizer_step >= args.preflight_max_optimizer_steps
                ):
                    break
                lr = _learning_rate(optimizer_step, total_steps, args.learning_rate, warmup_steps, args.lr_schedule)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                task = epoch_training[index % len(epoch_training)]
                canonical_source, canonical_target = random.sample(epoch_training, 2)
                loss, similarity, inverse, canonical = objective(task, canonical_source, canonical_target)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                logical_step += 1; optimizer_step += 1
                detached = [float(v.detach()) for v in (loss, similarity, inverse, canonical)]
                values.append(detached)
                direction = "mr_from_ct" if task.fixed.modality == "mr" else "ct_from_mr"
                direction_counts[direction] += 1
                iteration_record = {
                    "epoch": epoch,
                    "loss": detached[0],
                    "mind_similarity": detached[1],
                    "inverse_consistency": detached[2],
                    "canonical_cycle_consistency": detached[3],
                    "learning_rate": float(lr),
                    "logical_step": logical_step,
                    "optimizer_step": optimizer_step,
                    "direction_mr_from_ct": float(direction == "mr_from_ct"),
                }
                with iteration_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(iteration_record) + "\n")
                log_iteration(writer, iteration_record, optimizer_step)
                net.clean()
            if not values:
                raise RuntimeError("No optimizer update was executed in this invocation.")
            net.eval(); validation_values = []
            with torch.no_grad():
                for index, task in enumerate(validation):
                    source = training[(2 * index) % len(training)]
                    target = training[(2 * index + 1) % len(training)]
                    validation_values.append(float(objective(task, source, target)[0].detach()))
                    net.clean()
            validation_loss = float(np.mean(validation_values))
            validation_history.append(validation_loss)
            convergence = _convergence_diagnostic(validation_history)
            improved = validation_loss < best
            bad_epochs = 0 if improved else bad_epochs + 1
            would_stop_early = bool(
                args.patience > 0 and epoch + 1 >= args.min_epochs and bad_epochs >= args.patience
            )
            record = {
                "epoch": epoch, "train_loss": float(np.mean([v[0] for v in values])),
                "validation_loss": validation_loss,
                "mind_similarity": float(np.mean([v[1] for v in values])),
                "inverse_consistency": float(np.mean([v[2] for v in values])),
                "canonical_cycle_consistency": float(np.mean([v[3] for v in values])),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "segmentation_labels_used": False,
                "logical_step": logical_step, "optimizer_step": optimizer_step,
                "unique_anchors": sampling["unique_anchors"],
                "logical_batches_configured": sampling["logical_batches"],
                "logical_batches_executed": len(values),
                "rotating_repeat": sampling["rotating_repeat"],
                "sampling_schedule_sha256": sampling.get("schedule_sha256"),
                "mr_from_ct_updates": direction_counts["mr_from_ct"],
                "ct_from_mr_updates": direction_counts["ct_from_mr"],
                "early_stop_bad_epochs": bad_epochs, "would_stop_early": would_stop_early,
            }
            if convergence["available"]:
                record.update({
                    "last20_validation_relative_slope": convergence["relative_slope_per_epoch"],
                    "last20_not_converged": convergence["not_converged_still_improving"],
                })
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            log_record(writer, record, epoch)
            state = {
                "schema": "m2m_registration_checkpoint_v1", "method": "m2m_reg",
                "epoch": epoch, "global_step": optimizer_step,
                "logical_step": logical_step, "optimizer_step": optimizer_step,
                "best_validation_loss": min(best, validation_loss),
                "model_state_dict": net.regis_net.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": {
                    "implementation": "manual", "name": args.lr_schedule, "last_optimizer_step": optimizer_step,
                    "total_optimizer_steps": total_steps, "warmup_optimizer_steps": warmup_steps,
                    "updates_per_logical": 1, "base_lr": args.learning_rate,
                },
                "scaler_state_dict": scaler.state_dict(), "amp_enabled": False,
                "model_shape_zyx": list(args.model_shape_zyx), "num_cano": "-1",
                "loss": {"similarity": "MIND-SSC radius=2 dilation=2", "lambda_inverse": args.lambda_inverse, "lambda_canonical": args.lambda_canonical},
                "selection_metric": "label_free_validation_all_loss",
                "segmentation_labels_used": False,
                "run_protocol": args.run_protocol,
                "run_config": _serializable_arguments(args),
                "early_stopping": {
                    "min_epochs": args.min_epochs, "patience": args.patience,
                    "bad_epochs": bad_epochs, "would_trigger": would_stop_early,
                    "applied": False, "budget_policy": f"fixed_{args.epochs}_budget",
                },
                "validation_history": validation_history,
                "convergence_diagnostic": convergence,
                "sampling_state": sampling,
                "rng_state": capture_rng_state(),
                "training_provenance": {
                    "manifest": str(args.manifest.resolve()),
                    "manifest_sha256": sha256_file(args.manifest.resolve()),
                    "repo": repo_identity,
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
        if frozen_protocol and not args.preflight_max_optimizer_steps:
            expected_steps = args.epochs * steps
            if (logical_step, optimizer_step) != (expected_steps, expected_steps):
                raise RuntimeError(
                    f"{args.run_protocol} ended with incorrect step counts: "
                    f"{(logical_step, optimizer_step)} != {(expected_steps, expected_steps)}."
                )
        writer.close()
    finally:
        sys.path = [entry for entry in sys.path if entry != str(repo)]


if __name__ == "__main__":
    main()
