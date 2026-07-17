# vmex package map

This directory contains the Python package.

- `core/`: the implementation — one concern per module, each header docstring
  naming its VMEC2000 counterpart (see `core/__init__.py` for the module map).
  Includes the `vmec` CLI (`core/cli.py`).
- `__init__.py`: the public API — version, JAX persistent-cache setup, and
  lazy exports (`VmecInput`, `solve`, `solve_multigrid`,
  `solve_free_boundary`, wout IO, plotting, Boozer, mgrid/coils, `optimize`,
  `implicit`, `errors`).
- `_compat.py`: JAX import/backend policy helpers (x64, persistent
  compilation-cache directory selection) shared by `__init__.py` and
  `core/solver.py`.
- `doctor.py`: `vmec --doctor` installation/JAX-backend diagnostics.
- `resources/`: tiny package-bundled inputs needed after `pip install`, such
  as the `vmec --test` quick-start deck. User-facing examples live in
  `examples/data/`.
