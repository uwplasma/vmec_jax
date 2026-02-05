# Validation

Scripts that compare `vmec_jax` kernels against bundled `wout_*.nc` reference data.

- `step10_getfsq_parity_cases.py`: Step-10 scalar parity report (`fsqr/fsqz/fsql`) for bundled cases.
- `vmec_forces_rz_kernel_report.py`: diagnostic report for the VMEC-style R/Z force kernels.
- `constraint_pipeline_report.py`: constraint-force pipeline diagnostics (`tcon`, `gcon`) for a given `wout`.
- `residual_decomposition_report.py`: component-wise residual norms and top `(m,n)` contributors.
- `force_residual_report.py`: end-to-end force residual report on a chosen case.
- `wout_roundtrip.py`: write+read a minimal `wout_*.nc` and compare.
- `bsub_parity_figures.py`, `bmag_parity_figures.py`, `bsup_parity_figures.py`: parity figures vs `wout`.

Most scripts write `.npz` artifacts into `examples/outputs/`.
