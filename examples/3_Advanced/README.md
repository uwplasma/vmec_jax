# 3_Advanced examples

These scripts are intended for advanced use:

- longer runs,
- optimization / inverse problems,
- performance profiling,
- experimental features.

Most scripts in this folder will likely require optional dependencies.

Notes:
- Use `--jit-grad` only once you’re iterating repeatedly with fixed shapes (it increases compile latency).
- For early solver experiments, try `--preconditioner mode_diag` or `--preconditioner radial_tridi`.
- For implicit differentiation experiments, see `03_implicit_fixed_boundary_sensitivity.py`.
- For force-like residual reports (Step-10 target), see `05_force_residual_report.py`.
