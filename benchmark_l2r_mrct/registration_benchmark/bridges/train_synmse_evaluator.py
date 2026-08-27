"""Train the official SynMSE evaluator without hard-coded paths or labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..contract import load_registration_manifest
from .common import capture_rng_state, checkpoint_state, load_image_tensor, restore_rng_state, warp
from .train_pairwise import _learning_rate
from ..provenance import git_identity, sha256_file
from ..tensorboard_logging import create_summary_writer, log_iteration, log_record
from ..checkpointing import save_training_checkpoints


def _synthetic_view(image, max_displacement, affine_noise, generator):
    b, _, d, h, w = image.shape
    identity = torch.eye(3, 4, device=image.device, dtype=image.dtype)[None].repeat(b, 1, 1)
    noise = torch.randn(identity.shape, device=image.device, dtype=image.dtype, generator=generator) * affine_noise
    noise[:, :, -1] *= 2.0
    affine_grid = F.affine_grid(identity + noise, image.shape, align_corners=True)
    value = F.grid_sample(image, affine_grid, mode="bilinear", padding_mode="border", align_corners=True)
    coarse = torch.randn((b, 3, 4, 4, 4), device=image.device, dtype=image.dtype, generator=generator)
    flow = F.interpolate(coarse, size=(d, h, w), mode="trilinear", align_corners=True)
    flow = flow / flow.square().mean(dim=(2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
    return warp(value, flow * max_displacement)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--generator-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--tensorboard-dir", required=True, type=Path,
        help="Absolute formal TensorBoard directory outside the benchmark result tree.",
    )
    parser.add_argument("--model-shape-zyx", required=True, nargs=3, type=int)
    parser.add_argument("--source-role", choices=["fixed", "moving", "both"], default="moving")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr-schedule", choices=["cosine", "poly", "constant"], default="cosine")
    parser.add_argument("--max-displacement", type=float, default=8.0)
    parser.add_argument("--affine-noise", type=float, default=0.025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if not args.generator_checkpoint.is_file():
        raise FileNotFoundError(args.generator_checkpoint)
    tasks = load_registration_manifest(args.manifest)
    training = [task for task in tasks if task.split == "train"]
    validation = [task for task in tasks if task.split in {"validation", "val"}]
    if not training or not validation:
        raise ValueError("SynMSE evaluator training requires train and validation image splits.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The released SynMSE CT->MR generator constructs CUDA networks and requires CUDA.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    repo = args.repo.resolve(); sys.path.insert(0, str(repo))
    try:
        from model.Eva_model import Evaluator
        from model.cyclegan_networks3D import define_G
        generator_model = define_G(
            input_nc=1, output_nc=1, ngf=64, netG="resnet_9blocks", norm="instance",
            use_dropout=False, init_type="normal", init_gain=0.02, gpu_ids=[],
        ).to(device)
        generator_model.load_state_dict(checkpoint_state(args.generator_checkpoint, device), strict=True)
        generator_model.eval()
        for parameter in generator_model.parameters():
            parameter.requires_grad_(False)
        evaluator = Evaluator(2, 1).to(device)
        optimizer = torch.optim.Adam(evaluator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        start_epoch, global_step, best = 0, 0, float("inf")
        if args.resume:
            state = torch.load(args.resume, map_location=device)
            if state.get("schema") != "synmse_evaluator_checkpoint_v1" or list(state.get("model_shape_zyx", [])) != list(args.model_shape_zyx):
                raise ValueError("Resume checkpoint schema/model shape mismatch.")
            recorded_manifest = dict(state.get("training_provenance", {})).get("manifest_sha256")
            if recorded_manifest and recorded_manifest != sha256_file(args.manifest.resolve()):
                raise ValueError("Resume checkpoint was created from a different training manifest.")
            if state.get("source_role") != args.source_role:
                raise ValueError("Resume evaluator checkpoint source-role mismatch.")
            recorded_generator = state.get("generator_checkpoint_sha256")
            if recorded_generator and recorded_generator != sha256_file(args.generator_checkpoint.resolve()):
                raise ValueError("Resume evaluator checkpoint uses a different CT->MR generator.")
            evaluator.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state.get("global_step", 0))
            best = float(state["best_validation_loss"])
            restore_rng_state(state.get("rng_state"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / "epoch_metrics.jsonl"
        writer = create_summary_writer(args.output_dir, log_dir=args.tensorboard_dir)
        def source_paths(split_tasks):
            fixed = sorted({task.fixed.path.resolve() for task in split_tasks})
            moving = sorted({task.moving.path.resolve() for task in split_tasks})
            if args.source_role == "fixed":
                return fixed
            if args.source_role == "moving":
                return moving
            return sorted(set(fixed + moving))

        training_sources = source_paths(training)
        validation_sources = source_paths(validation)
        steps = args.steps_per_epoch or len(training_sources)
        total_steps, warmup_steps = args.epochs * steps, args.warmup_epochs * steps

        def source_image(path):
            return load_image_tensor(path, device, args.model_shape_zyx)

        def objective(path, deformation_seed):
            image = source_image(path)
            torch_generator = torch.Generator(device=device).manual_seed(deformation_seed)
            first = _synthetic_view(image, args.max_displacement, args.affine_noise, torch_generator)
            second = _synthetic_view(image, args.max_displacement, args.affine_noise, torch_generator)
            with torch.no_grad():
                translated_first = generator_model(first * 2.0 - 1.0)
            target_error = second - first
            predicted_error = evaluator(torch.cat([translated_first, second * 2.0 - 1.0], dim=1))
            return F.l1_loss(predicted_error, target_error)

        for epoch in range(start_epoch, args.epochs):
            evaluator.train(); epoch_training = list(training_sources); random.shuffle(epoch_training); values = []
            for index in range(steps):
                lr = _learning_rate(global_step, total_steps, args.learning_rate, warmup_steps, args.lr_schedule)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss = objective(epoch_training[index % len(epoch_training)], args.seed + global_step)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); global_step += 1
                values.append(float(loss.detach()))
                log_iteration(
                    writer,
                    {
                        "total_loss": float(loss.detach()),
                        "evaluator_l1": float(loss.detach()),
                        "learning_rate": float(lr),
                        "logical_step": global_step,
                        "optimizer_step": global_step,
                    },
                    global_step,
                )
            evaluator.eval(); validation_values = []
            with torch.no_grad():
                for index, path in enumerate(validation_sources):
                    validation_values.append(float(objective(path, args.seed + 100000 + index).detach()))
            validation_loss = float(np.mean(validation_values))
            record = {
                "epoch": epoch, "train_l1": float(np.mean(values)), "validation_l1": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]), "segmentation_labels_used": False,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            log_record(writer, record, epoch)
            state = {
                "schema": "synmse_evaluator_checkpoint_v1", "epoch": epoch, "global_step": global_step,
                "best_validation_loss": min(best, validation_loss),
                "model_state_dict": evaluator.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
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
                    "lr_schedule": args.lr_schedule, "max_displacement": args.max_displacement,
                    "affine_noise": args.affine_noise, "source_role": args.source_role,
                    "seed": args.seed, "checkpoint_every": args.checkpoint_every,
                    "tensorboard_dir": str(args.tensorboard_dir.resolve()),
                    "independent_train_sources": len(training_sources),
                    "independent_validation_sources": len(validation_sources),
                },
                "generator_checkpoint": str(args.generator_checkpoint.resolve()), "generator_frozen": True,
                "generator_checkpoint_sha256": sha256_file(args.generator_checkpoint.resolve()),
                "model_shape_zyx": list(args.model_shape_zyx), "source_role": args.source_role,
                "synthetic_deformation": {"max_displacement": args.max_displacement, "affine_noise": args.affine_noise},
                "selection_metric": "synthetic_error_map_validation_l1", "segmentation_labels_used": False,
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
