# Validation

Scripts that compare `vmec_jax` kernels against bundled `wout_*.nc` reference data.

- `step10_getfsq_parity_cases.py`: Step-10 scalar parity report (`fsqr/fsqz/fsql`) for bundled cases (uses `wout` Nyquist `bsup*` to isolate force-kernel parity).
- `vmec_forces_rz_kernel_report.py`: diagnostic report for the VMEC-style R/Z force kernels.
- `constraint_pipeline_report.py`: constraint-force pipeline diagnostics (`tcon`, `gcon`) for a given `wout`.
- `residual_decomposition_report.py`: component-wise residual norms and top `(m,n)` contributors.
- `residual_compare_fields_report.py`: compare full-field vs reference-field residual contributions.
- `lasym_block_report.py`: symmetric vs asymmetric (`tomnsps`/`tomnspa`) block contributions.
- `lasym_mode_trace_report.py`: trace a specific `(m,n)` mode through asymmetric blocks.
- `force_residual_report.py`: end-to-end force residual report on a chosen case.
- `wout_roundtrip.py`: write+read a minimal `wout_*.nc` and compare.
- `bsub_parity_figures.py`, `bmag_parity_figures.py`, `bsup_parity_figures.py`: parity figures vs `wout`.
- `external_vmec_driver_compare.py`: run VMEC2000 or VMEC++ (if installed) and compare the resulting `wout` to bundled references; optionally computes vmec_jax B-field parity metrics.
- `n3are_vmecpp_stage_diagnostics.py`: stage-by-stage diagnostics for the `vmecpp_iter` path on n3are (geometry, tomnsps block norms, force scalars, VMEC-grid `|B|`) for initial guess vs post-solver state.
- `vmecpp_stage_parity_pipeline.py`: VMEC++-run, stage-by-stage parity report that identifies the first failing block (`geometry -> bsup -> bsub -> getfsq`).
- `vmecpp_stage_parity_pipeline.py` currently uses a looser `bsub` stage threshold (`4e-2`) so diagnostics continue into `getfsq`/solver-update mismatches instead of stopping on the known few-`1e-2` `bsubu` gap.
- `vmecpp_stage_parity_pipeline.py` now defaults to the input-grid angular resolution for `getfsq` parity; pass `--hi-res` only for exploratory field diagnostics.
- `vmecpp_stage_parity_pipeline.py` now uses direct Nyquist Fourier evaluation for `use_wout_bsup` reference fields, avoiding a small VMEC-synthesis mismatch in output `bsup*`.
- `vmecpp_stage_parity_pipeline.py` now passes VMEC input constraints (`indata/TCON0`) into the self-consistency force path; on n3are this clears the `getfsq` stage gate (`first_failing_stage=none`) and leaves solver-update parity as the next mismatch.
- `vmecpp_getfsq_decomposition.py`: sweeps `getfsq` conventions (`include_edge`, `scalxc`, `m=1`) on a VMEC++ final state to isolate residual-scalar convention mismatches.
- `vmecpp_bsub_metric_probe.py`: decomposes `bsub` parity on a VMEC++ final state and attributes the remaining gap to metric pathways (`guu/guv/gvv`) and `bsup` terms.
- `vmecpp_jxbout_compare.py`: compares vmec_jax `bcovar` fields against VMEC++ `jxbout` internal arrays using VMEC internal-grid ordering (`ns, nzeta, ntheta_eff`), and reports `wout`-evaluation baselines to separate kernel mismatch from output-format mismatch.
- `n3are_vmec_vs_vmecjax.py`: side-by-side VMEC2000 vs vmec_jax plots with optional `--solve` execution (moved from `visualization/`).
  - For the current parity stage, `--no-solve` is the recommended visualization baseline; the fixed-boundary update loop is still being tightened.
  - With `--solve --solver vmecpp_iter`, iteration traces now include `fsqr/fsqz/fsql`, preconditioned `fsqr1/fsqz1/fsql1`, `dt_eff/update_rms`, and `step_status` for update-loop diagnostics.

Notes:
- `residual_decomposition_report.py` and `residual_compare_fields_report.py` now support reference-field kernels that expose a minimal `bc` object by falling back to the `wout`-based force normalization path.

Most scripts write `.npz` artifacts into `examples/outputs/`.
