"""Three-GPU DDP training for the V4-final full-resolution descriptor."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data import (
    Anchor,
    deterministic_foreground_crop,
    distributed_epoch_anchors,
    load_anchor_inventory,
    prepare_anchor,
)
from .model import V4FinalModel, representation_objective
from .protocol import V4FinalProtocol
from .state import (
    atomic_save,
    canonical_hash,
    capture_rng,
    load_checkpoint,
    make_checkpoint,
    restore_rng,
    source_hash,
)


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _seed(seed: int, rank: int) -> None:
    selected = int(seed) + int(rank) * 10_007
    random.seed(selected)
    np.random.seed(selected)
    torch.manual_seed(selected)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(selected)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _mean_across_ranks(metrics: Mapping[str, torch.Tensor], world_size: int) -> Dict[str, float]:
    names = tuple(sorted(metrics))
    values = torch.stack([metrics[name].detach().float() for name in names])
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size
    return {name: float(value.cpu()) for name, value in zip(names, values)}


def _validation_mean(
    model,
    anchors: Sequence[Anchor],
    protocol: V4FinalProtocol,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    model.eval()
    sums = torch.zeros(5, device=device, dtype=torch.float64)
    count = torch.zeros(1, device=device, dtype=torch.float64)
    selected = anchors[: min(len(anchors), protocol.training.validation_volumes)]
    with torch.no_grad():
        for index in range(rank, len(selected), world_size):
            anchor = selected[index]
            volume = prepare_anchor(
                anchor, ct_min_hu=protocol.inference.ct_foreground_min_hu
            )
            image, mask = deterministic_foreground_crop(
                volume,
                protocol.training.crop_shape_dzyx,
                seed=protocol.training.seed + 50_000_000 + index,
                minimum_foreground_fraction=protocol.training.minimum_foreground_fraction,
            )
            image = image[None].to(device, non_blocking=True)
            mask = mask[None].to(device, non_blocking=True)
            generator = torch.Generator(device=device).manual_seed(
                protocol.training.seed + 60_000_000 + index
            )
            with _autocast(device, protocol.training.precision):
                result = representation_objective(
                    model,
                    image,
                    mask,
                    protocol.training,
                    task="appearance",
                    generator=generator,
                )
            row = torch.stack(
                [
                    result.metrics["total"],
                    result.metrics["contrastive"],
                    result.metrics["variance"],
                    result.metrics["positive_cosine"],
                    result.metrics["top1"],
                ]
            ).double()
            sums += row
            count += 1
    if world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if int(count.item()) < 1:
        raise RuntimeError("No validation anchors were evaluated.")
    values = sums / count
    return {
        name: float(value.cpu())
        for name, value in zip(
            ("total", "contrastive", "variance", "positive_cosine", "top1"), values
        )
    }


def _gather_rng(rank: int, world_size: int):
    state = capture_rng()
    if world_size == 1:
        return (state,)
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(state, gathered, dst=0)
    return tuple(gathered) if rank == 0 else ()


def _disk_guard(path: Path, minimum_gib: float) -> None:
    free = shutil.disk_usage(path).free / (1024.0**3)
    if free < float(minimum_gib):
        raise OSError(f"Only {free:.2f} GiB free; checkpoint floor is {minimum_gib:.2f} GiB.")


def train_descriptor(
    protocol: V4FinalProtocol,
    *,
    manifest: Union[str, Path],
    output_dir: Union[str, Path],
    tensorboard_dir: Union[str, Path],
    resume: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    max_epochs_this_invocation: Optional[int] = None,
) -> Mapping[str, object]:
    rank, local_rank, world_size = _distributed()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        selected_device = torch.device("cuda", local_rank)
    else:
        if world_size != 1:
            raise ValueError("Multi-process V4-final training requires CUDA/NCCL.")
        selected_device = torch.device(device)
    _seed(protocol.training.seed, rank)
    manifest_path = Path(manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    tensorboard = Path(tensorboard_dir).expanduser().resolve()
    inventory = load_anchor_inventory(manifest_path)
    if len(inventory.train) % world_size != 0:
        raise ValueError(
            f"Formal DDP requires exact no-repeat sharding: {len(inventory.train)} anchors / "
            f"{world_size} GPUs is not integral."
        )
    local_steps_per_epoch = len(inventory.train) // world_size
    if local_steps_per_epoch < 1:
        raise ValueError("More DDP ranks than training anchors.")
    protocol_dict = protocol.to_dict()
    if rank == 0:
        if resume is None and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Fresh V4-final output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        tensorboard.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    raw_model = V4FinalModel(protocol.descriptor).to(selected_device)
    train_model = (
        DistributedDataParallel(raw_model, device_ids=[local_rank], output_device=local_rank)
        if world_size > 1
        else raw_model
    )
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=protocol.training.learning_rate)
    total_optimizer_steps = protocol.training.epochs * local_steps_per_epoch
    warmup_steps = protocol.training.warmup_epochs * local_steps_per_epoch

    def learning_rate_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_optimizer_steps - warmup_steps - 1, 1)
        progress = min(1.0, max(0.0, progress))
        floor = protocol.training.minimum_learning_rate_factor
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    scaler = torch.cuda.amp.GradScaler(
        enabled=selected_device.type == "cuda" and protocol.training.precision == "fp16"
    )
    start_epoch = 0
    logical_step = 0
    optimizer_step = 0
    best_validation = float("inf")
    if resume is not None:
        payload = load_checkpoint(resume, map_location="cpu")
        if payload["protocol_sha256"] != canonical_hash(protocol_dict):
            raise ValueError("Resume protocol differs from the checkpoint.")
        if payload["source_sha256"] != source_hash():
            raise ValueError("Resume source tree differs from the checkpoint.")
        if payload["manifest_sha256"] != inventory.manifest_sha256:
            raise ValueError("Resume manifest differs from the checkpoint.")
        if int(payload["world_size"]) != world_size:
            raise ValueError("Resume world size differs; optimizer-step semantics would change.")
        raw_model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        restore_rng(payload["rng_by_rank"][rank])
        start_epoch = int(payload["epoch"]) + 1
        logical_step = int(payload["logical_step"])
        optimizer_step = int(payload["optimizer_step"])
        best_validation = float(payload["best_validation"])

    run_manifest = {
        "schema": "pracm_v4_final_training_run_v1",
        "status": "running",
        "protocol": protocol_dict,
        "protocol_sha256": canonical_hash(protocol_dict),
        "source_sha256": source_hash(),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": inventory.manifest_sha256,
        "train_anchors": len(inventory.train),
        "validation_anchors": len(inventory.validation),
        "world_size": world_size,
        "per_gpu_batch": 1,
        "logical_anchors_per_epoch": len(inventory.train),
        "optimizer_steps_per_epoch": local_steps_per_epoch,
        "tensorboard_dir": str(tensorboard),
        "labels_used": False,
        "public8_used": False,
        "checkpoint_selection": "label-free validation representation loss",
        "main_inference_solver": "descriptor ConvexAdam",
    }
    run_manifest_path = output / "run_manifest.json"
    if rank == 0:
        if resume is not None and run_manifest_path.is_file():
            previous = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if previous.get("tensorboard_dir") != str(tensorboard):
                raise ValueError("Resume TensorBoard path differs from the locked run.")
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tensorboard), max_queue=1000, flush_secs=120)
    metrics_path = output / "epoch_metrics.jsonl"
    iteration_path = output / "iteration_metrics.jsonl"
    invocation_stop = protocol.training.epochs
    if max_epochs_this_invocation is not None:
        if max_epochs_this_invocation < 1:
            raise ValueError("max_epochs_this_invocation must be positive.")
        invocation_stop = min(invocation_stop, start_epoch + int(max_epochs_this_invocation))

    completed_epochs = start_epoch
    for epoch in range(start_epoch, invocation_stop):
        local_anchors, repeats = distributed_epoch_anchors(
            inventory.train,
            epoch=epoch,
            seed=protocol.training.seed,
            rank=rank,
            world_size=world_size,
        )
        if repeats != 0 or len(local_anchors) != local_steps_per_epoch:
            raise RuntimeError("Formal V4-final epoch unexpectedly repeated an anchor.")
        train_model.train()
        epoch_sums = {name: 0.0 for name in ("total", "contrastive", "variance", "positive_cosine", "top1")}
        for local_index, anchor in enumerate(local_anchors):
            volume = prepare_anchor(
                anchor, ct_min_hu=protocol.inference.ct_foreground_min_hu
            )
            anchor_seed = (
                protocol.training.seed
                + epoch * 10_000_019
                + rank * 100_003
                + local_index
            )
            image, mask = deterministic_foreground_crop(
                volume,
                protocol.training.crop_shape_dzyx,
                seed=anchor_seed,
                minimum_foreground_fraction=protocol.training.minimum_foreground_fraction,
            )
            image = image[None].to(selected_device, non_blocking=True)
            mask = mask[None].to(selected_device, non_blocking=True)
            generator = torch.Generator(device=selected_device).manual_seed(anchor_seed + 70_000_000)
            task_value = ((anchor_seed * 2_654_435_761) % 1_000_000) / 1_000_000.0
            task = "geometry" if task_value < protocol.training.geometry_probability else "appearance"
            optimizer.zero_grad(set_to_none=True)
            with _autocast(selected_device, protocol.training.precision):
                result = representation_objective(
                    train_model,
                    image,
                    mask,
                    protocol.training,
                    task=task,
                    generator=generator,
                )
            if not torch.isfinite(result.loss):
                raise FloatingPointError(
                    f"Non-finite V4-final loss at epoch={epoch}, optimizer_step={optimizer_step}."
                )
            scaler.scale(result.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), protocol.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            logical_step += world_size
            optimizer_step += 1
            reduced = _mean_across_ranks(
                {key: value for key, value in result.metrics.items() if key in epoch_sums},
                world_size,
            )
            for name in epoch_sums:
                epoch_sums[name] += reduced[name]
            if rank == 0 and optimizer_step % protocol.training.iteration_log_every_optimizer_steps == 0:
                record = {
                    "epoch": epoch,
                    "logical_step": logical_step,
                    "optimizer_step": optimizer_step,
                    "task_rank0": task,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **reduced,
                }
                with iteration_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                assert writer is not None
                for name, value in reduced.items():
                    writer.add_scalar(f"iteration/{name}", value, optimizer_step)
                writer.add_scalar(
                    "iteration/logical_step", logical_step, optimizer_step
                )
                writer.add_scalar(
                    "iteration/learning_rate", optimizer.param_groups[0]["lr"], optimizer_step
                )
        train_mean = {name: value / local_steps_per_epoch for name, value in epoch_sums.items()}
        validation = _validation_mean(
            train_model,
            inventory.validation,
            protocol,
            device=selected_device,
            rank=rank,
            world_size=world_size,
        )
        improved = validation["total"] < best_validation
        if improved:
            best_validation = validation["total"]
        rng_by_rank = _gather_rng(rank, world_size)
        if rank == 0:
            _disk_guard(output, protocol.training.minimum_free_disk_gib)
            payload = make_checkpoint(
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                protocol=protocol_dict,
                epoch=epoch,
                logical_step=logical_step,
                optimizer_step=optimizer_step,
                best_validation=best_validation,
                manifest_path=manifest_path,
                manifest_sha256=inventory.manifest_sha256,
                tensorboard_dir=tensorboard,
                world_size=world_size,
                rng_by_rank=rng_by_rank,
            )
            checkpoints = output / "checkpoints"
            atomic_save(payload, checkpoints / "last.pt")
            if improved:
                atomic_save(payload, checkpoints / "best.pt")
            if (
                (epoch + 1) % protocol.training.checkpoint_every_epochs == 0
                and epoch + 1 < protocol.training.epochs
            ):
                atomic_save(payload, checkpoints / f"epoch_{epoch + 1:04d}.pt")
            epoch_record = {
                "epoch": epoch,
                "logical_step": logical_step,
                "optimizer_step": optimizer_step,
                "train": train_mean,
                "validation": validation,
                "best_validation": best_validation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(epoch_record, sort_keys=True) + "\n")
            assert writer is not None
            for name, value in train_mean.items():
                writer.add_scalar(f"train/{name}", value, epoch)
            for name, value in validation.items():
                writer.add_scalar(f"validation/{name}", value, epoch)
            writer.add_scalar("optimization/logical_step", logical_step, epoch)
            writer.add_scalar("optimization/optimizer_step", optimizer_step, epoch)
            writer.flush()
        if world_size > 1:
            dist.barrier()
        completed_epochs = epoch + 1

    complete = completed_epochs >= protocol.training.epochs
    if rank == 0:
        run_manifest.update(
            {
                "status": "completed" if complete else "paused",
                "completed_epochs": completed_epochs,
                "logical_step": logical_step,
                "optimizer_step": optimizer_step,
                "best_validation": best_validation,
                "training_complete": complete,
            }
        )
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert writer is not None
        writer.close()
    if world_size > 1:
        dist.barrier()
    return {
        "rank": rank,
        "world_size": world_size,
        "completed_epochs": completed_epochs,
        "logical_step": logical_step,
        "optimizer_step": optimizer_step,
        "best_validation": best_validation,
        "training_complete": complete,
    }


__all__ = ["train_descriptor"]

