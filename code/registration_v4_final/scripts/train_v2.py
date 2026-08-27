"""Authoritative three-GPU V4-final faithful representation training."""

from __future__ import annotations

import argparse
import json

from ..protocol import load_protocol
from ..training_runtime_v2 import train_descriptor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs-this-invocation", type=int)
    args = parser.parse_args(argv)
    result = train_descriptor(
        load_protocol(args.config),
        manifest=args.manifest,
        output_dir=args.output_dir,
        tensorboard_dir=args.tensorboard_dir,
        resume=args.resume,
        device=args.device,
        max_epochs_this_invocation=args.max_epochs_this_invocation,
    )
    if int(result["rank"]) == 0:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

