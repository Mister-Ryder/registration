# Method version branches

This repository deliberately keeps the three scientific method identities on
separate branches. Datasets, full server exports, temporary probes and large
archives are excluded from Git.

| Branch | Method identity | Main package |
|---|---|---|
| `registration_v4_pracm` | Original PRA-CM V4 development line | `code/registration_v4_pracm` |
| `registration_v4_core` | Frozen faithful DSIR / V4-Core line | `code/registration_v4_final` |
| `registration_v5` | Consolidated DINO-anchor + dense-corefix + capture-router | `code/registration_v5` |

The V4-Core and V5 branches also preserve the frozen V4-Core best checkpoint
and its SHA-256 identity. V5 keeps the official DINO-Reg and ConvexAdam sources
as external/upstream dependencies; no foundation-model weight is committed.

The `main` branch is only the branch index. It does not represent an experiment.
