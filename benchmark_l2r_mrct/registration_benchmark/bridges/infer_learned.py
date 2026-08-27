"""Unified label-free inference for released learned comparison models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .common import checkpoint_state, load_pair_tensors, resize_flow_to_native


def _synmse(args, fixed, moving, device):
    sys.path.insert(0, str(args.repo))
    try:
        from model.Reg_model import VxmDense
        cfg = {
            "unet_features": [[16, 32, 32, 64, 64], [64, 64, 32, 32, 16, 16]],
            "diff": int(args.diffeomorphic),
        }
        model = VxmDense(cfg=cfg).to(device)
        model.load_state_dict(checkpoint_state(args.checkpoint, device), strict=True)
        model.eval()
        with torch.no_grad():
            # Match the released SynMSE training/evaluator intensity domain.
            return model(moving * 2.0 - 1.0, fixed * 2.0 - 1.0)
    finally:
        sys.path.pop(0)


def _dgmir(args, fixed, moving, device):
    if device.type != "cuda":
        raise RuntimeError("The released DGMIR implementation constructs CUDA kernels and requires CUDA.")
    sys.path.insert(0, str(args.repo))
    try:
        from model2 import DGMIR
        model = DGMIR(list(fixed.shape[-3:])).to(device)
        model.load_state_dict(checkpoint_state(args.checkpoint, device), strict=True)
        model.eval()
        with torch.no_grad():
            flow, _ = model(fixed, moving, "test")
        return flow
    finally:
        sys.path.pop(0)


def _m2m_family(args, fixed, moving, device):
    sys.path.insert(0, str(args.repo))
    try:
        from models import make_network
        from icon_registration import config as icon_config
        icon_config.device = device
        model_name = {"m2m_reg": "gradicon", "transmorph_mind": "transmorph", "corrmlp_mind": "corrmlp"}[args.method]
        namespace = SimpleNamespace(
            model=model_name,
            input_shape=tuple(fixed.shape),
            num_cano="0",
            lambda_inv=0.5,
            lambda_can=0.1,
            log_mono=False,
            small=bool(args.small),
        )
        net = make_network(namespace, include_last_step=False, use_label=False)
        net.regis_net.load_state_dict(checkpoint_state(args.checkpoint, device), strict=True)
        net = net.to(device).eval()
        with torch.no_grad():
            net.identity_map.isIdentity = True
            mapping = net.regis_net(moving, fixed)
            coordinates = mapping(net.identity_map)
            normalized_delta = coordinates - net.identity_map
            factors = torch.as_tensor([max(v - 1, 1) for v in fixed.shape[-3:]], device=device, dtype=coordinates.dtype)
            flow = normalized_delta * factors[None, :, None, None, None]
        return flow
    finally:
        sys.path.pop(0)


def _capacity_control(args, fixed, moving, device):
    if args.method == "transmorph_mind":
        source = args.repo / "TransMorph"
        sys.path.insert(0, str(source))
        try:
            from models.TransMorph import CONFIGS, TransMorph
            config = CONFIGS["TransMorph"]
            config.img_size = tuple(fixed.shape[-3:])
            model = TransMorph(config).to(device)
            model.load_state_dict(checkpoint_state(args.checkpoint, device), strict=True)
            model.eval()
            with torch.no_grad():
                _, flow = model(torch.cat([moving, fixed], dim=1))
            return flow
        finally:
            sys.path.pop(0)
    source = args.repo / "CorrMLP"
    sys.path.insert(0, str(source))
    try:
        from networks import CorrMLP
        model = CorrMLP(use_checkpoint=True).to(device)
        model.load_state_dict(checkpoint_state(args.checkpoint, device), strict=True)
        model.eval()
        with torch.no_grad():
            _, flow = model(fixed, moving)
        return flow
    finally:
        sys.path.pop(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["synmse", "dgmir_u", "m2m_reg", "transmorph_mind", "corrmlp_mind"])
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument("--moving", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-shape-zyx", nargs=3, type=int)
    parser.add_argument("--diffeomorphic", action="store_true")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device)
    model_shape = tuple(args.model_shape_zyx or ())
    if args.method == "synmse":
        # A formal SynMSE checkpoint records the grid used in training.  Use it
        # by default and resize the resulting voxel displacement to the native
        # fixed grid below.  This keeps inference consistent and avoids a
        # silent change from the trained grid to an arbitrary native shape.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if not model_shape and isinstance(checkpoint, dict):
            model_shape = tuple(int(value) for value in checkpoint.get("model_shape_zyx", ()))
        if isinstance(checkpoint, dict):
            architecture = dict(checkpoint.get("architecture", {}))
            if architecture and int(architecture.get("diff", 0)) != int(args.diffeomorphic):
                raise ValueError(
                    "SynMSE --diffeomorphic does not match the registration checkpoint architecture."
                )
    fixed_volume, _, fixed, moving = load_pair_tensors(args.fixed, args.moving, device, model_shape)
    if args.method == "synmse":
        flow = _synmse(args, fixed, moving, device)
    elif args.method == "dgmir_u":
        flow = _dgmir(args, fixed, moving, device)
    elif args.method == "m2m_reg":
        flow = _m2m_family(args, fixed, moving, device)
    else:
        flow = _capacity_control(args, fixed, moving, device)
    native = resize_flow_to_native(flow, tuple(reversed(fixed_volume.shape_xyz)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, native, allow_pickle=False)


if __name__ == "__main__":
    main()
