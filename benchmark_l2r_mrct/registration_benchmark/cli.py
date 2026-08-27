"""Unified command line for dataset contracts, inference, evaluation, and DNS training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import (
    MULTIPHASE_GROUPS, build_l2r_public_manifests, build_l2r_unpaired_training_manifest,
    build_multiphase_manifests, build_plcr_manifests,
)
from .runner import run_manifest
from .preflight import check_method


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regbench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-multiphase", help="Build image-only and evaluation-only manifests.")
    build.add_argument("--inventory", required=True, type=Path)
    build.add_argument("--registration-output", required=True, type=Path)
    build.add_argument("--evaluation-output", required=True, type=Path)
    build.add_argument("--training-output", type=Path, help="Optional all-four-phase train/validation manifest.")
    build.add_argument("--dataset-id", required=True)
    build.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    build.add_argument("--evaluation-splits", nargs="+", default=["test"])

    plcr = commands.add_parser(
        "build-plcr",
        help="Audit the frozen PLC-R 172/29/49 split and build formal train/test manifests.",
    )
    plcr.add_argument("--inventory", required=True, type=Path)
    plcr.add_argument("--registration-output", required=True, type=Path)
    plcr.add_argument("--evaluation-output", required=True, type=Path)
    plcr.add_argument("--training-output", required=True, type=Path)
    plcr.add_argument("--dataset-id", default="PLC-R-250")
    plcr.add_argument("--task-groups", nargs="+", choices=sorted(MULTIPHASE_GROUPS), default=["precontrast_target"])
    build.add_argument("--task-groups", nargs="+", choices=list(MULTIPHASE_GROUPS), default=["precontrast_target"])
    l2r = commands.add_parser("build-l2r-public8", help="Convert the frozen public-8 inventory to isolated manifests.")
    l2r.add_argument("--inventory", required=True, type=Path)
    l2r.add_argument("--registration-output", required=True, type=Path)
    l2r.add_argument("--evaluation-output", required=True, type=Path)
    l2r.add_argument("--dataset-id", default="L2R-MRCT-P8")
    l2r.add_argument("--dataset-root", type=Path)
    l2r_train = commands.add_parser("build-l2r-unpaired-train", help="Build deterministic MR<-CT pairs from unpaired images.")
    l2r_train.add_argument("--inventory", required=True, type=Path)
    l2r_train.add_argument("--output", required=True, type=Path)
    l2r_train.add_argument("--dataset-root", type=Path)
    l2r_train.add_argument("--dataset-id", default="L2R-MRCT-unpaired90")
    l2r_train.add_argument("--validation-count", type=int, default=5)
    l2r_train.add_argument("--seed", type=int, default=2024)

    run = commands.add_parser("run", help="Run any method through the canonical pair interface.")
    run.add_argument("--pair-manifest", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--pair-id")
    run.add_argument("--splits", nargs="+")
    run.add_argument("--task-groups", nargs="+")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Evaluate completed flows in an isolated label process.")
    evaluate.add_argument("--pair-manifest", required=True, type=Path)
    evaluate.add_argument("--evaluation-manifest", required=True, type=Path)
    evaluate.add_argument("--results-dir", required=True, type=Path)
    evaluate.add_argument("--method-ids", nargs="+", required=True)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    train_dns = commands.add_parser("train-dns", help="Run the paper reproduction's unlabeled DNS training.")
    train_dns.add_argument("args", nargs=argparse.REMAINDER)
    preflight = commands.add_parser("preflight", help="Check one method environment without running a case.")
    preflight.add_argument("--config", required=True, type=Path)
    plan = commands.add_parser("plan-experiments", help="Write the frozen per-method plan for PLC-R or L2R-MRCT.")
    plan.add_argument("--dataset", required=True, choices=["plcr", "l2r_mrct"])
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--policy", type=Path)
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "build-multiphase":
        pair_count, label_count = build_multiphase_manifests(
            args.inventory, args.registration_output, args.evaluation_output,
            dataset_id=args.dataset_id, splits=args.splits, task_groups=args.task_groups,
            training_output=args.training_output, evaluation_splits=args.evaluation_splits,
        )
        result = {"registration_pairs": pair_count, "evaluation_pairs": label_count, "task_groups": args.task_groups}
    elif args.command == "build-plcr":
        result = build_plcr_manifests(
            args.inventory,
            args.registration_output,
            args.evaluation_output,
            args.training_output,
            dataset_id=args.dataset_id,
            task_groups=args.task_groups,
        )
    elif args.command == "build-l2r-public8":
        pair_count, label_count = build_l2r_public_manifests(
            args.inventory, args.registration_output, args.evaluation_output,
            dataset_id=args.dataset_id, dataset_root=args.dataset_root,
        )
        result = {"registration_pairs": pair_count, "evaluation_pairs": label_count, "protocol": "fixed MR <- moving CT"}
    elif args.command == "build-l2r-unpaired-train":
        train_count, validation_count = build_l2r_unpaired_training_manifest(
            args.inventory, args.output, dataset_id=args.dataset_id, dataset_root=args.dataset_root,
            validation_count=args.validation_count, seed=args.seed,
        )
        result = {"training_pairs": train_count, "validation_pairs": validation_count, "pairing": "unpaired_cross_product"}
    elif args.command == "run":
        result = run_manifest(
            args.pair_manifest, args.config, args.output_dir, pair_id=args.pair_id,
            splits=set(args.splits or ()), task_groups=set(args.task_groups or ()),
            resume=args.resume, continue_on_error=args.continue_on_error,
        )
    elif args.command == "evaluate":
        from .evaluation import evaluate_outputs
        result = evaluate_outputs(
            args.pair_manifest, args.evaluation_manifest, args.results_dir, args.method_ids, args.output_dir
        )
    elif args.command == "train-dns":
        from .dns.training import main as train_dns_main
        train_dns_main(args.args); result = {"training": "completed"}
    elif args.command == "plan-experiments":
        from .experiment_policy import write_execution_plan
        result = write_execution_plan(args.output, args.dataset, policy_path=args.policy)
    else:
        result = check_method(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.command == "preflight" and not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
