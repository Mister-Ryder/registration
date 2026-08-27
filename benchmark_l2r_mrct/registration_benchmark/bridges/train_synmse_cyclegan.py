"""Train the SynMSE prerequisite CT->MR CycleGAN on unlabeled L2R volumes.

The released SynMSE repository requires users to pre-train this model but
hard-codes the authors' private checkpoint path.  This bridge retains the
released 3-D ResNet generators/discriminators and standard CycleGAN objective,
while supplying benchmark manifests, deterministic validation, provenance,
checkpointing, and TensorBoard logging.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..contract import load_registration_manifest
from ..provenance import git_identity, sha256_file
from ..tensorboard_logging import create_summary_writer, log_iteration, log_record
from ..checkpointing import save_training_checkpoints
from .common import capture_rng_state, load_image_tensor, restore_rng_state
from .train_pairwise import _learning_rate


def _set_grad(networks, enabled: bool) -> None:
    for network in networks:
        for parameter in network.parameters():
            parameter.requires_grad_(enabled)


def _pool_state(pool) -> dict:
    """Capture the official replay buffer so an interrupted run is resumable."""

    return {
        "pool_size": int(pool.pool_size),
        "num_imgs": int(getattr(pool, "num_imgs", 0)),
        "images": [image.detach().cpu() for image in getattr(pool, "images", [])],
    }


def _restore_pool(pool, state: Optional[dict], device: torch.device) -> None:
    if not state:
        return
    if int(state.get("pool_size", -1)) != int(pool.pool_size):
        raise ValueError("Resume CycleGAN replay-pool size mismatch.")
    images = list(state.get("images", []))
    num_imgs = int(state.get("num_imgs", len(images)))
    if num_imgs != len(images) or num_imgs > pool.pool_size:
        raise ValueError("Resume CycleGAN replay-pool state is inconsistent.")
    pool.num_imgs = num_imgs
    pool.images = [image.to(device) for image in images]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--tensorboard-dir", required=True, type=Path,
        help="Absolute formal TensorBoard directory outside the benchmark result tree.",
    )
    parser.add_argument("--model-shape-zyx", required=True, nargs=3, type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--lr-schedule", choices=["cosine", "poly", "constant"], default="constant")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)

    tasks = load_registration_manifest(args.manifest)
    training = [task for task in tasks if task.split == "train"]
    validation = [task for task in tasks if task.split in {"validation", "val"}]
    if not training or not validation:
        raise ValueError("CycleGAN requires non-empty train and label-free validation splits.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The released SynMSE 3-D CycleGAN constructs CUDA networks and requires CUDA.")
    torch.cuda.set_device(0 if device.index is None else device.index)
    repo = args.repo.resolve(); sys.path.insert(0, str(repo))
    try:
        from model.cycle_gan_model import ImagePool
        from model.cyclegan_networks3D import GANLoss, define_D, define_G

        # Domain A is moving CT; domain B is fixed MR.  The exported G_A is
        # therefore exactly the CT->MR generator consumed by SynMSE evaluator training.
        generator_ct_to_mr = define_G(1, 1, args.ngf, "resnet_9blocks", "instance", False, "normal", 0.02, []).to(device)
        generator_mr_to_ct = define_G(1, 1, args.ngf, "resnet_9blocks", "instance", False, "normal", 0.02, []).to(device)
        discriminator_mr = define_D(1, args.ndf, "basic", 3, "instance", False, "normal", 0.02, []).to(device)
        discriminator_ct = define_D(1, args.ndf, "basic", 3, "instance", False, "normal", 0.02, []).to(device)
        gan = GANLoss(use_lsgan=True).to(device)
        generators = [generator_ct_to_mr, generator_mr_to_ct]
        discriminators = [discriminator_mr, discriminator_ct]
        optimizer_g = torch.optim.Adam(
            list(generator_ct_to_mr.parameters()) + list(generator_mr_to_ct.parameters()),
            lr=args.learning_rate, betas=(0.5, 0.999),
        )
        optimizer_d = torch.optim.Adam(
            list(discriminator_mr.parameters()) + list(discriminator_ct.parameters()),
            lr=args.learning_rate, betas=(0.5, 0.999),
        )
        pool_mr, pool_ct = ImagePool(args.pool_size), ImagePool(args.pool_size)
        start_epoch, global_step, best = 0, 0, float("inf")
        if args.resume:
            state = torch.load(args.resume, map_location=device)
            if state.get("schema") != "synmse_cyclegan_checkpoint_v1" or list(state.get("model_shape_zyx", [])) != list(args.model_shape_zyx):
                raise ValueError("Resume CycleGAN checkpoint schema/model shape mismatch.")
            recorded_manifest = dict(state.get("training_provenance", {})).get("manifest_sha256")
            if recorded_manifest and recorded_manifest != sha256_file(args.manifest.resolve()):
                raise ValueError("Resume checkpoint was created from a different training manifest.")
            networks = state["networks"]
            generator_ct_to_mr.load_state_dict(networks["generator_ct_to_mr"], strict=True)
            generator_mr_to_ct.load_state_dict(networks["generator_mr_to_ct"], strict=True)
            discriminator_mr.load_state_dict(networks["discriminator_mr"], strict=True)
            discriminator_ct.load_state_dict(networks["discriminator_ct"], strict=True)
            optimizer_g.load_state_dict(state["optimizer_g"]); optimizer_d.load_state_dict(state["optimizer_d"])
            replay_pools = dict(state.get("replay_pools", {}))
            _restore_pool(pool_mr, replay_pools.get("mr"), device)
            _restore_pool(pool_ct, replay_pools.get("ct"), device)
            start_epoch, global_step = int(state["epoch"]) + 1, int(state.get("global_step", 0))
            best = float(state["best_validation_cycle_l1"]); restore_rng_state(state.get("rng_state"))

        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / "epoch_metrics.jsonl"
        writer = create_summary_writer(args.output_dir, log_dir=args.tensorboard_dir)
        def domain_paths(split_tasks):
            mr = sorted({task.fixed.path.resolve() for task in split_tasks})
            ct = sorted({task.moving.path.resolve() for task in split_tasks})
            return mr, ct

        training_mr, training_ct = domain_paths(training)
        validation_mr, validation_ct = domain_paths(validation)
        steps = args.steps_per_epoch or max(len(training_mr), len(training_ct))
        total_steps, warmup_steps = args.epochs * steps, args.warmup_epochs * steps

        def images(mr_path, ct_path):
            mr = load_image_tensor(mr_path, device, args.model_shape_zyx)
            ct = load_image_tensor(ct_path, device, args.model_shape_zyx)
            return ct * 2.0 - 1.0, mr * 2.0 - 1.0

        def generator_terms(real_ct, real_mr):
            fake_mr = generator_ct_to_mr(real_ct); recovered_ct = generator_mr_to_ct(fake_mr)
            fake_ct = generator_mr_to_ct(real_mr); recovered_mr = generator_ct_to_mr(fake_ct)
            adversarial = gan(discriminator_mr(fake_mr), True) + gan(discriminator_ct(fake_ct), True)
            cycle = F.l1_loss(recovered_ct, real_ct) + F.l1_loss(recovered_mr, real_mr)
            identity = F.l1_loss(generator_ct_to_mr(real_mr), real_mr) + F.l1_loss(generator_mr_to_ct(real_ct), real_ct)
            total = adversarial + args.lambda_cycle * cycle + args.lambda_cycle * args.lambda_identity * identity
            return total, adversarial, cycle, identity, fake_mr, fake_ct

        def discriminator_term(network, real, fake):
            return 0.5 * (gan(network(real), True) + gan(network(fake.detach()), False))

        for epoch in range(start_epoch, args.epochs):
            for network in generators + discriminators:
                network.train()
            # Cycle the independently shuffled modality domains.  The 1,748
            # cross-product manifest entries are not 1,748 independent cases.
            epoch_mr, epoch_ct = list(training_mr), list(training_ct)
            random.shuffle(epoch_mr); random.shuffle(epoch_ct)
            values = []
            for index in range(steps):
                lr = _learning_rate(global_step, total_steps, args.learning_rate, warmup_steps, args.lr_schedule)
                for optimizer in (optimizer_g, optimizer_d):
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                real_ct, real_mr = images(
                    epoch_mr[index % len(epoch_mr)], epoch_ct[index % len(epoch_ct)]
                )
                _set_grad(discriminators, False); optimizer_g.zero_grad(set_to_none=True)
                loss_g, adversarial, cycle, identity, fake_mr, fake_ct = generator_terms(real_ct, real_mr)
                loss_g.backward(); optimizer_g.step()
                _set_grad(discriminators, True); optimizer_d.zero_grad(set_to_none=True)
                loss_d_mr = discriminator_term(discriminator_mr, real_mr, pool_mr.query(fake_mr))
                loss_d_ct = discriminator_term(discriminator_ct, real_ct, pool_ct.query(fake_ct))
                loss_d = loss_d_mr + loss_d_ct; loss_d.backward(); optimizer_d.step()
                values.append([float(value.detach()) for value in (loss_g, loss_d, adversarial, cycle, identity)])
                global_step += 1
                log_iteration(
                    writer,
                    {
                        "total_loss": float((loss_g + loss_d).detach()),
                        "generator_loss": float(loss_g.detach()),
                        "discriminator_loss": float(loss_d.detach()),
                        "adversarial_loss": float(adversarial.detach()),
                        "cycle_l1": float(cycle.detach()),
                        "identity_l1": float(identity.detach()),
                        "learning_rate": float(lr),
                        "logical_step": global_step,
                        "optimizer_step_generator": global_step,
                        "optimizer_step_discriminator": global_step,
                    },
                    global_step,
                )

            for network in generators:
                network.eval()
            validation_cycle = []
            with torch.no_grad():
                validation_steps = max(len(validation_mr), len(validation_ct))
                for index in range(validation_steps):
                    real_ct, real_mr = images(
                        validation_mr[index % len(validation_mr)],
                        validation_ct[index % len(validation_ct)],
                    )
                    recovered_ct = generator_mr_to_ct(generator_ct_to_mr(real_ct))
                    recovered_mr = generator_ct_to_mr(generator_mr_to_ct(real_mr))
                    validation_cycle.append(float((F.l1_loss(recovered_ct, real_ct) + F.l1_loss(recovered_mr, real_mr)).detach()))
            validation_loss = float(np.mean(validation_cycle))
            record = {
                "epoch": epoch, "generator_loss": float(np.mean([v[0] for v in values])),
                "discriminator_loss": float(np.mean([v[1] for v in values])),
                "adversarial_loss": float(np.mean([v[2] for v in values])),
                "cycle_l1": float(np.mean([v[3] for v in values])),
                "identity_l1": float(np.mean([v[4] for v in values])),
                "validation_cycle_l1": validation_loss,
                "learning_rate": float(optimizer_g.param_groups[0]["lr"]),
                "segmentation_labels_used": False,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            log_record(writer, record, epoch)
            state = {
                "schema": "synmse_cyclegan_checkpoint_v1", "epoch": epoch, "global_step": global_step,
                "best_validation_cycle_l1": min(best, validation_loss),
                # Generic key is deliberately the released CT->MR generator so
                # train_synmse_evaluator can consume this formal checkpoint directly.
                "model_state_dict": generator_ct_to_mr.state_dict(),
                "networks": {
                    "generator_ct_to_mr": generator_ct_to_mr.state_dict(),
                    "generator_mr_to_ct": generator_mr_to_ct.state_dict(),
                    "discriminator_mr": discriminator_mr.state_dict(),
                    "discriminator_ct": discriminator_ct.state_dict(),
                },
                "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
                "replay_pools": {"mr": _pool_state(pool_mr), "ct": _pool_state(pool_ct)},
                "logical_step": global_step,
                "optimizer_steps": {"generator": global_step, "discriminator": global_step},
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
                    "lr_schedule": args.lr_schedule, "ngf": args.ngf, "ndf": args.ndf,
                    "lambda_cycle": args.lambda_cycle, "lambda_identity": args.lambda_identity,
                    "pool_size": args.pool_size, "seed": args.seed,
                    "checkpoint_every": args.checkpoint_every,
                    "tensorboard_dir": str(args.tensorboard_dir.resolve()),
                    "independent_train_mr": len(training_mr),
                    "independent_train_ct": len(training_ct),
                    "independent_validation_mr": len(validation_mr),
                    "independent_validation_ct": len(validation_ct),
                },
                "model_shape_zyx": list(args.model_shape_zyx), "domain_a": "ct", "domain_b": "mr",
                "objective": {"gan": "least_squares", "lambda_cycle": args.lambda_cycle, "lambda_identity": args.lambda_identity},
                "segmentation_labels_used": False, "selection_metric": "label_free_validation_cycle_l1",
                "rng_state": capture_rng_state(),
                "training_provenance": {
                    "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest.resolve()),
                    "repo": git_identity(repo),
                },
            }
            improved = validation_loss < best
            save_training_checkpoints(
                state, args.output_dir, epoch=epoch, improved=improved,
                # ``last.pt`` already is the complete final-epoch recovery
                # point; do not duplicate a multi-GB CycleGAN payload there.
                checkpoint_every=0 if epoch + 1 == args.epochs else args.checkpoint_every,
            )
            if improved:
                best = validation_loss
        writer.close()
    finally:
        sys.path = [entry for entry in sys.path if entry != str(repo)]


if __name__ == "__main__":
    main()
