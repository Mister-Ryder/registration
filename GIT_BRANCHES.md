# Method version branches

This repository deliberately keeps the three scientific method identities on
separate branches. Datasets, full server exports, temporary probes and large
archives are excluded from Git.

| Branch | Method identity | Main package |
|---|---|---|
| `registration_v4_pracm` | Original PRA-CM V4 development line | `code/registration_v4_pracm` |
| `registration_v4_core` | Frozen faithful DSIR / V4-Core line | `code/registration_v4_final` |
| `registration_v5` | Consolidated DINO-anchor + dense-corefix + capture-router | `code/registration_v5` |
| `results_l2r_protocol300` | Frozen Learn2Reg MR–CT B00–B12 comparison results | `results/L2R_MRCT_protocol300_20260825` |

The V4-Core and V5 branches also preserve the frozen V4-Core best checkpoint
and its SHA-256 identity. V5 keeps the official DINO-Reg and ConvexAdam sources
as external/upstream dependencies; no foundation-model weight is committed.

The `main` branch is only the branch index. It does not represent an experiment.
The `results_l2r_protocol300` branch is a result archive, not a method implementation;
its lightweight files are intended for review while the complete local server archive
remains under `results/L2R_MRCT_protocol300_20260825`.
