# PRA-CM V4-final

This is an isolated replacement for the unsuccessful V4 A/B/Full branch.  It
does not load those checkpoints.  Its single publication-facing path is:

1. paper-aligned Figure-9 feature extraction at input resolution;
2. direct plus dilated DNS and the 24-channel feature-squeezing DSIR;
3. raw, modality-aware foreground support before normalization;
4. descriptor-agnostic ConvexAdam: discrete correlation, coupled convex
   regularization, inverse consistency and Adam refinement.

The dense normalized-grid IO implementation remains in
`registration_v4_pracm.inference.instance_optimization` only as a diagnostic
control.  It is not the default solver here.

Training uses label-free standalone images.  Each anchor alternates between
faithful nonlinear-intensity invariance and known-diffeomorphic equivariance.
On three GPUs, 84 unique anchors form 28 optimizer steps per epoch; checkpoints
record logical anchors and optimizer steps separately.  Best/last checkpoints
contain model, optimizer, scheduler, AMP scaler, all-rank RNG, protocol/source
hashes and manifest identity.  TensorBoard writes are iteration-indexed and
buffered for 120 seconds.

Formal server paths are intentionally supplied by the launcher rather than
embedded in the Python package.  The fixed locations are:

- source: `/root/autodl-tmp/l2r_benchmark_v1/code_fixes/v4_final_20260826`
- run: `/root/autodl-tmp/l2r_benchmark_v1/runs/v4_final_protocol300/B10_v4_final`
- TensorBoard: `/root/tf-logs/v4_final_protocol300/B10_v4_final`
- results: `/root/autodl-tmp/l2r_benchmark_v1/results/v4_final_protocol300`
- evaluation: `/root/autodl-tmp/l2r_benchmark_v1/evaluation/v4_final_protocol300/B10_v4_final`

