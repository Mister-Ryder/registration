"""Train PRA-CM on PLC-R phases or unpaired L2R MR/CT domains."""

from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..training.engine import run_training


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tensorboard-dir",
        required=True,
        help="Absolute formal TensorBoard directory outside the checkpoint tree.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument(
        "--max-epochs-this-invocation",
        type=int,
        help="Optional safe segment length; the checkpoint retains the full configured schedule.",
    )
    args = parser.parse_args(argv)
    result = run_training(
        load_config(args.config),
        output_dir=args.output_dir,
        tensorboard_dir=args.tensorboard_dir,
        device=args.device,
        resume=args.resume,
        max_epochs_this_invocation=args.max_epochs_this_invocation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
