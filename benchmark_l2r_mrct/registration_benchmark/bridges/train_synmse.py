"""Label-isolated SynMSE registration training with a frozen official evaluator."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from ..contract import load_registration_manifest
from .common import capture_rng_state, checkpoint_state, gradient_loss, load_pair_tensors, restore_rng_state, warp
from .train_pairwise import _learning_rate
from ..provenance import git_identity, sha256_file
from ..tensorboard_logging import create_summary_writer, log_iteration, log_record
from ..checkpointing import save_training_checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluator-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--tensorboard-dir", required=True, type=Path,
        help="Absolute formal TensorBoard directory outside the benchmark result tree.",
    )
    parser.add_argument("--model-shape-zyx", required=True, nargs=3, type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr-schedule", choices=["cosine", "poly", "constant"], default="cosine")
    parser.add_argument("--similarity-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.5)
    parser.add_argument("--diffeomorphic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)

    tasks = load_registration_manifest(args.manifest)
    training = [task for task in tasks if task.split == "train"]
    validation = [task for task in tasks if task.split in {"validation", "val"}]
    if not training or not validation:
        raise ValueError("SynMSE requires non-empty train and label-free validation splits.")
    if not args.evaluator_checkpoint.is_file():
        raise FileNotFoundError(args.evaluator_checkpoint)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    repo = args.repo.resolve(); sys.path.insert(0, str(repo))
    try:
        from model.Eva_model import Evaluator
        from model.Reg_model import VxmDense
        cfg = {
            "unet_features": [[16, 32, 32, 64, 64], [64, 64, 32, 32, 16, 16]],
            "diff": int(args.diffeomorphic),
        }
        registration = VxmDense(cfg=cfg).to(device)
        evaluator = Evaluator(2, 1).to(device)
        evaluator.load_state_dict(checkpoint_state(args.evaluator_checkpoint, device), strict=True)
        evaluator.eval()
        for parameter in evaluator.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(registration.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        start_epoch, global_step, best = 0, 0, float("inf")
        if args.resume:
            state = torch.load(args.resume, map_location=device)
            if state.get("method") != "synmse" or list(state.get("model_shape_zyx", [])) != list(args.model_shape_zyx):
                raise ValueError("Resume checkpoint method/model shape mismatch.")
            recorded_manifest = dict(state.get("training_provenance", {})).get("manifest_sha256")
            if recorded_manifest and recorded_manifest != sha256_file(args.manifest.resolve()):
                raise ValueError("Resume checkpoint was created from a different training manifest.")
            recorded_evaluator = state.get("evaluator_checkpoint_sha256")
            if recorded_evaluator and recorded_evaluator != sha256_file(args.evaluator_checkpoint.resolve()):
                raise ValueError("Resume registration checkpoint uses a different SynMSE evaluator.")
            architecture = dict(state.get("architecture", {}))
            if architecture and int(architecture.get("diff", 0)) != int(args.diffeomorphic):
                raise ValueError("Resume registration checkpoint diffeomorphic-setting mismatch.")
            registration.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state.get("global_step", 0))
            best = float(state["best_validation_loss"])
            restore_rng_state(state.get("rng_state"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / "epoch_metrics.jsonl"
        writer = create_summary_writer(args.output_dir, log_dir=args.tensorboard_dir)
        steps = args.steps_per_epoch or len(training)
        total_steps, warmup_steps = args.epochs * steps, args.warmup_epochs * steps

        def loss_for(task):
            _, _, fixed, moving = load_pair_tensors(
                task.fixed.path, task.moving.path, device, args.model_shape_zyx
            )
            # The released SynMSE data pipeline normalizes both modalities to
            # [-1, 1].  The evaluator is trained in exactly that intensity
            # domain (translated MR-style CT first, CT second), so registration
            # must not feed the benchmark loader's native [0, 1] tensors.
            fixed = fixed * 2.0 - 1.0
            moving = moving * 2.0 - 1.0
            flow = registration(moving, fixed)
            warped = warp(moving, flow)
            error_map = evaluator(torch.cat([fixed, warped], dim=1)).abs()
            similarity = error_map.mean()
            smoothness = gradient_loss(flow)
            return args.similarity_weight * similarity + args.smoothness_weight * smoothness, similarity, smoothness

        for epoch in range(start_epoch, args.epochs):
            registration.train(); epoch_training = sorted(training, key=lambda value: value.pair_id); random.shuffle(epoch_training); train_values = []
            for index in range(steps):
                lr = _learning_rate(global_step, total_steps, args.learning_rate, warmup_steps, args.lr_schedule)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss, similarity, smoothness = loss_for(epoch_training[index % len(epoch_training)])
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); global_step += 1
                train_values.append([float(loss.detach()), float(similarity.detach()), float(smoothness.detach())])
                log_iteration(
                    writer,
                    {
                        "total_loss": float(loss.detach()),
                        "synmse_error": float(similarity.detach()),
                        "smoothness": float(smoothness.detach()),
                        "learning_rate": float(lr),
                        "logical_step": global_step,
                        "optimizer_step": global_step,
                    },
                    global_step,
                )
            registration.eval(); validation_values = []
            with torch.no_grad():
                for task in validation:
                    validation_values.append(float(loss_for(task)[0].detach()))
            validation_loss = float(np.mean(validation_values))
            record = {
                "epoch": epoch, "train_loss": float(np.mean([v[0] for v in train_values])),
                "validation_loss": validation_loss,
                "synmse_error": float(np.mean([v[1] for v in train_values])),
                "smoothness": float(np.mean([v[2] for v in train_values])),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "segmentation_labels_used": False,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            log_record(writer, record, epoch)
            state = {
                "schema": "synmse_registration_checkpoint_v1", "method": "synmse",
                "epoch": epoch, "global_step": global_step,
                "best_validation_loss": min(best, validation_loss),
                "model_state_dict": registration.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "logical_step": global_step, "optimizer_step": global_step,
                "lr_scheduler": {
                    "implementation": "closed_form_per_step", "kind": args.lr_schedule,
                    "warmup_epochs": args.warmup_epochs, "total_epochs": args.epochs,
                    "steps_per_epoch": steps, "total_steps": total_steps,
                },
                "amp": {"enabled": False, "scaler_state_dict": None},
                "early_stopping": {"enabled": False, "state": None},
                "training_config": {
                    "epochs": args.epochs, "steps_per_epoch": steps,
                    "learning_rate": args.learning_rate, "warmup_epochs": args.warmup_epochs,
                    "lr_schedule": args.lr_schedule, "similarity_weight": args.similarity_weight,
                    "smoothness_weight": args.smoothness_weight,
                    "diffeomorphic": bool(args.diffeomorphic), "seed": args.seed,
                    "checkpoint_every": args.checkpoint_every,
                    "tensorboard_dir": str(args.tensorboard_dir.resolve()),
                },
                "architecture": cfg, "model_shape_zyx": list(args.model_shape_zyx),
                "evaluator_checkpoint": str(args.evaluator_checkpoint.resolve()),
                "evaluator_checkpoint_sha256": sha256_file(args.evaluator_checkpoint.resolve()),
                "evaluator_frozen": True, "segmentation_labels_used": False,
                "selection_metric": "label_free_validation_synmse_plus_smoothness",
                "rng_state": capture_rng_state(),
                "training_provenance": {
                    "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest.resolve()),
                    "repo": git_identity(repo),
                },
            }
            improved = validation_loss < best
            save_training_checkpoints(
                state, args.output_dir, epoch=epoch, improved=improved,
                checkpoint_every=0 if epoch + 1 == args.epochs else args.checkpoint_every,
            )
            if improved:
                best = validation_loss
        writer.close()
    finally:
        sys.path = [entry for entry in sys.path if entry != str(repo)]


if __name__ == "__main__":
    main()
