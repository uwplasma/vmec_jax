Code structure
==============

Top-level package layout (selected):

- ``vmec_jax/namelist.py``: minimal Fortran namelist reader (``&INDATA``)
- ``vmec_jax/config.py``: run discretization config extracted from inputs
- ``vmec_jax/static.py``: static grids + Fourier basis tensors (PyTrees)
- ``vmec_jax/state.py``: coefficient containers (PyTrees)
- ``vmec_jax/coords.py`` / ``vmec_jax/geom.py``: geometry and metric kernels
- ``vmec_jax/field.py`` / ``vmec_jax/energy.py``: B-field and energy diagnostics
- ``vmec_jax/solve.py``: early solver prototypes (fixed-boundary)
- ``vmec_jax/wout.py``: minimal ``wout_*.nc`` reader for regression

The ``examples/`` folder contains a small set of user-facing scripts and curated
figure / parity demos. Developer-only diagnostics and research scripts live
under ``tools/``. See ``examples/README.md`` for the current layout.

For most scripts, the recommended import surface is the small public API in
``vmec_jax.api``.
