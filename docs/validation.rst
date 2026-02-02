Validation and regression testing
=================================

``vmec-jax`` is developed using a regression-first workflow: each porting step
introduces a small kernel and validates it against VMEC2000 outputs (typically
via ``wout_*.nc``).

Bundled regression cases
------------------------

The repo includes several small, low-resolution reference cases used in examples
and tests:

- 3D stellarator (vacuum):
  - input: ``examples/input.LandremanSenguptaPlunk_section5p3_low_res``
  - reference output: ``examples/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc``

- Tokamak sanity cases (vacuum):
  - ``examples/input.circular_tokamak`` + ``examples/wout_circular_tokamak_reference.nc``
  - ``examples/input.up_down_asymmetric_tokamak`` + ``examples/wout_up_down_asymmetric_tokamak_reference.nc``

- Finite-beta case:
  - ``examples/input.li383_low_res`` + ``examples/wout_li383_low_res_reference.nc``

What is validated today
-----------------------

The tests in ``tests/`` cover:

- correct INDATA parsing,
- boundary evaluation and agreement with the ``s=1`` state surface,
- metric/Jacobian positivity and shape checks,
- stepwise regressions vs ``wout`` (Nyquist ``sqrt(g)``, ``bsup*``, scalar integrals).
- Step-10 parity scaffolding for VMEC-style ``forces``/``tomnsps``/``getfsq`` (currently marked ``xfail`` while parity is still being completed).

Running tests::

  pytest -q

If you do not have ``netCDF4`` installed, tests that require ``wout`` I/O will be
skipped.
