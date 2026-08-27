# V4-Core frozen baseline (2026-08-27)

This directory is an immutable source snapshot of the first successful V4-Core
release. Do not edit files under this directory. All subsequent descriptor/DNS
work must be implemented in `code/registration_v4_dnsfix` (or a later,
separately named release directory).

## Verified public-8 result

- Mean Dice: `0.6633615210`
- ASSD: `8.234422 mm`
- HD95: `27.164855 mm`
- Fold fraction: `5.9382e-05`
- Training completed: epoch `299/299`, logical step `25200`, optimizer step `8400`
- Best validation checkpoint: epoch `217`, validation loss `0.40073484`

## Frozen source composition

- `code/registration_v4_final`
- `code/registration_v4_final_release`
- `code/registration_v4_pracm`
- `benchmark_l2r_mrct/registration_benchmark`
- `benchmark_l2r_mrct/server/v4_final_20260826`

## Canonical remote artifacts (server port 46608)

- Run: `/root/autodl-tmp/l2r_benchmark_v1/runs/v4_final_faithful_protocol300/B10_v4_final`
- Best checkpoint: `/root/autodl-tmp/l2r_benchmark_v1/runs/v4_final_faithful_protocol300/B10_v4_final/train/checkpoints/best.pt`
- Last checkpoint: `/root/autodl-tmp/l2r_benchmark_v1/runs/v4_final_faithful_protocol300/B10_v4_final/train/checkpoints/last.pt`
- Flows: `/root/autodl-tmp/l2r_benchmark_v1/results/v4_final_faithful_protocol300/B10_v4_final`
- Evaluation: `/root/autodl-tmp/l2r_benchmark_v1/evaluation/v4_final_faithful_protocol300/B10_v4_final`
- TensorBoard: `/root/tf-logs/v4_final_faithful_protocol300/B10_v4_final`

`SOURCE_SHA256SUMS.csv` records the exact hash of every frozen local source file.
