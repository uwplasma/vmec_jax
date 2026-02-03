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
- Step-10 parity scaffolding for VMEC-style ``forces``/``tomnsps``/``getfsq``.
- VMEC convention checks used by Step-10 kernels (e.g. ``chipf -> chips`` inversion and ``equif`` normalization parity).
- an early end-to-end regression that a Gauss-Newton residual solver decreases a VMEC-style residual objective on ``input.circular_tokamak`` (this is *not* yet a full VMEC2000 equilibrium-parity solve).

Step-10 parity status (fsqr/fsqz/fsql)
--------------------------------------------------

The current Step-10 parity regression compares the scalar residuals computed by
``vmec-jax`` against those stored in bundled VMEC2000 ``wout_*.nc`` reference
files, using the same internal VMEC angle grid conventions:

.. math::

   \texttt{bcovar} \rightarrow \texttt{forces} \rightarrow \texttt{tomnsps} \rightarrow \texttt{getfsq}.

The scoreboard below reports relative errors
:math:`|\hat f - f|/\max(|f|,\epsilon)` for each scalar:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18

   * - Case
     - fsqr rel. err
     - fsqz rel. err
     - fsql rel. err
   * - circular_tokamak
     - ~4.9e-2
     - ~4.6e-2
     - ~4.8e-3
   * - up_down_asymmetric_tokamak
     - ~4.1e-2
     - ~1.3e-2
     - ~3.2e-2
   * - li383_low_res
     - ~1.6e-1
     - ~1.2e-1
     - ~1.1e-1
   * - LandremanSenguptaPlunk_section5p3_low_res
     - ~1.1e-1
     - ~8.9e-2
     - ~3.2e-1

Notes:

- The largest current mismatch is ``fsql`` on the 3D ``LandremanSenguptaPlunk`` case;
  this is treated as the main “scoreboard” item for further Step-10 parity work.
- These numbers are expected to change as parity improves; the authoritative
  regression is ``tests/test_step10_residue_getfsq_parity.py``.

Running tests::

  pytest -q

If you do not have ``netCDF4`` installed, tests that require ``wout`` I/O will be
skipped.

Optional: validating against a local VMEC2000 build
---------------------------------------------------

In this workspace layout (``VMEC2000`` checked out next to ``vmec_jax_git``), the
test suite includes a small smoke test that calls the VMEC2000 Python API to run
a low-resolution case and compares the produced ``wout_*.nc`` to the bundled
reference.

There is also an optional integration parity test that runs VMEC2000, reads the
generated ``wout_*.nc``, and checks that ``vmec-jax`` reproduces the Step-10 scalar
residuals (``fsqr/fsqz/fsql``) on the same case.

This test is skipped automatically if:

- the ``VMEC2000`` folder is not present,
- or the VMEC2000 Python extension has not been built under
  ``VMEC2000/_skbuild/*/cmake-install/python``.

It requires ``mpi4py`` (VMEC2000's wrapper expects MPI to be initialized even in
single-process mode).

To run the integration tests locally::

  VMEC2000_INTEGRATION=1 pytest -q -m vmec2000
