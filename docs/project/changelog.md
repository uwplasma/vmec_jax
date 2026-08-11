# Changelog

## 0.5.0

VMEX 0.5 makes converged equilibria optimizer-neutral building blocks:

- `VmecProblem` exposes consistent SciPy and JAX residual, Jacobian, scalar,
  and value/gradient contracts for QI, QS, geometry, profile, stability, and
  user-authored objectives.
- Exact implicit derivatives, automatic Jacobian direction, hot and
  perturbation restarts, ESS scales, bounded Jacobian batches, and smooth
  failed-trial handling are production defaults.
- `forward_ftol`, `forward_max_iterations`, and `max_fsq_ratio` make forward
  convergence and derivative certification explicit. Evaluations report FSQ
  and certification diagnostics.
- J-invariant QI and maximum-J objectives can share one Boozer transform;
  free-boundary, virtual-casing, bootstrap, stability, plotting, and standard
  wout workflows remain available.
- Automatic ensemble and finite-difference worker counts respect process CPU
  limits on shared, containerized, and scheduler-managed machines.

Upgrade notes:

- Tuple weights use cost semantics in `VmecProblem.from_tuples`: weight `w`
  contributes `w * (f - target)^2 / 2`. The compatibility
  `optimize.least_squares` wrapper retains its documented residual-weight
  convention.
- The experimental cross-iterate `recycle=True` Jacobian lane was removed;
  it was slower than the certified block/GMRES defaults. The production
  reverse adjoint continues to use GCROT internally.
- Decorative optimization scans and duplicate ESS examples were removed.
  Use `benchmarks/optimization.py` and the backend-specific QI examples.

Before tagging `v0.5.0`, the release PR must pass the PR gate, the bounded
Nightly physics campaigns, documentation link checking, an sdist/wheel build,
and installation/import tests of both artifacts. Publishing a GitHub release
whose tag matches `pyproject.toml` triggers the trusted PyPI workflow.

Earlier release notes live on [GitHub Releases](https://github.com/uwplasma/vmex/releases).
