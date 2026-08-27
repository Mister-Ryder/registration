# Registration V5

`registration_v5` is the consolidated, publication-facing implementation of
the best completed MR--CT result in this workspace.  It is not another
experimental delta directory.  All V5-owned code is contained here:

1. the faithful full-resolution 4-channel feature / 24-channel DSIR descriptor
   and its label-free protocol-300 training runtime;
2. the audited dense-corefix solver (spacing-12 capture, spacing-6 residual,
   coupled forward/backward consistency, pullback composition and
   Jacobian-safe acceptance);
3. the label-free capture-range router and canonical flow materializer;
4. the data, DNS, spatial, checkpoint and instance-optimization primitives
   needed by those components.

The default anchor is the pinned **official B04 DINO-Reg** model.  Its upstream
repository and DINOv2 weights remain external dependencies because they are
not V5-owned code.  V5 never describes this anchor as a newly trained V5
network.  No Python module is imported from an experimental
`registration_v4_*` directory.

## Frozen result identity

- public-8 canonical-v2: Mean Dice `0.7859497458400165`, ASSD
  `4.5772762260810245` mm, HD95 `19.858701211148382` mm, fold fraction
  `0.026441595289442274`;
- dense-corefix selected for `0004` and `0014`;
- byte-identical DINO anchor selected for the other six cases;
- router uses no segmentation labels, organ scores or case-specific constants;
- frozen V4-Core best checkpoint SHA-256:
  `6ba1c54ab260f4fb830b019caeaaf8414c1b45aae435adb4ff9eb68592d5bb70`.

This is the current best completed method, but it is below both 0.80 and the
project target of 0.85.  The manifest records that limitation rather than
relabeling the result.

## Exact information flow

```text
MR/CT pair
  |-- official DINO-Reg -----------------------------> u_DINO
  `-- frozen V4-Core DSIR -> dense-corefix ----------> u_dense + QA
                                                        |
                         p95(coarse)>24 and topology/FB safe?
                                      | yes                    | no
                                      v                        v
                                  u_dense                  u_DINO
                                      `------ byte copy -------'
```

Every flow is fixed/output-grid to moving/input sampling displacement,
`[3,D,H,W]`, component order `dz,dy,dx`, units fixed-grid voxels.

## Formal pair execution

Put the workspace `code` directory on `PYTHONPATH`.  Run the two frozen
candidates independently, then route them.  Separate processes are recommended
so the two foundation/descriptor models do not retain GPU memory together.

```bash
python -m registration_v5.scripts.dino_anchor \
  --fixed MR.nii.gz --moving CT.nii.gz --pair-id 0004 \
  --repo benchmark_l2r_mrct/third_party/dino_reg \
  --work-dir work/0004/dino --output-flow work/0004/dino_flow.npz

python -m registration_v5.scripts.infer_dense \
  --config code/registration_v5/configs/registration_v5_frozen_v4core_protocol300.yaml \
  --checkpoint releases/v4_core_0p663361_frozen_20260827/remote_artifacts/server_46608/root/autodl-tmp/l2r_benchmark_v1/runs/v4_final_faithful_protocol300/B10_v4_final/train/checkpoints/best.pt \
  --fixed MR.nii.gz --moving CT.nii.gz \
  --output-flow work/0004/dense_flow.npz \
  --output-qa work/0004/dense_flow.qa.json

python -m registration_v5.scripts.route_pair \
  --pair-id 0004 --dino-flow work/0004/dino_flow.npz \
  --corefix-flow work/0004/dense_flow.npz \
  --corefix-qa work/0004/dense_flow.qa.json \
  --output-dir results/registration_v5/0004 --method-id registration_v5
```

`route_pair` refuses to overwrite an existing output directory, validates both
candidate flow contracts, checks the dense solver QA identity, copies the
selected flow without reserialization, and verifies its SHA-256 afterward.

## Training entry

The consolidated faithful descriptor training code remains available through
`python -m registration_v5.scripts.train_descriptor`.  The config above intentionally
keeps the frozen V4-Core protocol ID so its checkpoint contract can be audited.
A newly trained V5 checkpoint must use a new protocol ID and must not overwrite
the frozen checkpoint or be reported under the frozen result manifest.

## Package map

- `dns/`, `network.py`, `training/`: descriptor and label-free training;
- `solver/corefix.py`: dense nonrigid solver;
- `routing/capture.py`: capture/topology/FB decision;
- `scripts/dino_anchor.py`: pinned official DINO anchor bridge;
- `scripts/infer_dense.py`: frozen descriptor + corefix inference;
- `scripts/route_pair.py`: immutable whole-case arbitration;
- `VERSION_MANIFEST.json`: exact method/result identity;
- `tests/`: component and release-contract tests.
