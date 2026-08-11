One solver iteration
====================

Each VMEX iteration follows VMEC2000's ``funct3d`` ordering exactly:
synthesize the geometry from the spectral state, form the fields and forces,
project the residuals back to Fourier space, and advance with a damped
second-order Richardson step. This page documents the discretization, the
update, the time-step controller, the restart rules, and the two execution
lanes; the preconditioners that make each step effective are in
:doc:`preconditioners`.

Discretization summary
----------------------

Radial grid
~~~~~~~~~~~

A uniform grid in :math:`s \in [0,1]` with ``ns`` points:

.. math::

   s_j = \frac{j}{ns-1},\qquad j=0,\dots,ns-1.

Following VMEC2000, quantities live on a mix of the *full mesh* (:math:`s_j`)
and the *half mesh* (:math:`s_{j-1/2}`): geometry derivatives, the Jacobian,
and ``|B|``-type quantities are half-mesh; R/Z coefficients and ``iotaf`` are
full-mesh. Odd-m coefficients are stored internally with the axis-regular
:math:`\sqrt{s}` factor removed (``scalxc``, see
:doc:`spectral-representation`), and R/Z and :math:`\lambda`
evolution starts from the m-dependent ``jmin2``/``jlam`` radial indices
(``vmec_params.f``). These conventions are implemented in
:mod:`vmex.core.geometry` and :mod:`vmex.core.setup`.

Angular grids and Fourier transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uniform tensor-product grids in :math:`\theta` and :math:`\zeta` (one field
period) with VMEC's symmetry-reduced theta extent. The analysis transform
(``tomnsps``) is the two-stage weighted DFT of
:doc:`spectral-representation`, built from precomputed trig tables
(``fixaray.f``) in :mod:`vmex.core.fourier`; the transforms
(``totzsps/totzspa`` synthesis, ``tomnsps/tomnspa`` analysis) are batched
``dot_general`` matmuls in :mod:`vmex.core.transforms` — GEMM-friendly
and XLA-fusable while matching VMEC2000 normalization exactly.

Geometry pipeline
~~~~~~~~~~~~~~~~~

Per iteration (:mod:`vmex.core.geometry`, VMEC2000 ``jacobian.f``):

1. synthesize :math:`(R,Z,\lambda)` and their angular derivatives on the
   ``(s,\theta,\zeta)`` grid from the spectral state;
2. form radial derivatives on the half mesh;
3. compute the half-mesh Jacobian :math:`\sqrt{g}` and the metric elements
   ``guu, guv, gvv``;
4. evaluate the Jacobian sign proxy :math:`\tau`; a sign change away from the
   axis flags a bad Jacobian (``irst = 2``).

The Richardson update
---------------------

VMEC solves the fixed-boundary equilibrium by evolving the stacked Fourier
coefficients :math:`\mathbf{x}` of :math:`(R,Z,\lambda)` with a
preconditioned, damped second-order Richardson iteration:

.. math::

   \mathbf{v}_{k+1} = \frac{1-d_k}{1+d_k}\,\mathbf{v}_k
                      + \frac{\Delta t}{1+d_k}\,P^{-1}\mathbf{r}(\mathbf{x}_k),
   \qquad
   \mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t\,\mathbf{v}_{k+1},

where :math:`\mathbf{r}` is the spectral force residual and the damping

.. math::

   d_k = \tfrac{1}{2}\,\Delta t\,\langle \mathrm{otau}\rangle,
   \qquad
   \mathrm{otau} \leftarrow \min\!\left(\left|\log\frac{\mathrm{fsq}_k}{\mathrm{fsq}_{k-1}}\right| / \Delta t,\; \frac{0.15}{\Delta t}\right)

is averaged over the last ``ndamp = 10`` steps (``evolve.f``). This is
implemented in :mod:`vmex.core.step`:
:func:`~vmex.core.step.damping_coefficients` advances the ``ndamp``
window and returns the ``(b1, fac)`` pair,
:func:`~vmex.core.step.momentum_update` applies the velocity/position
update, and the traced controller scalars (``delt``, damping history,
best-residual trackers, ``iter1``, ``ijacob``) live in
:class:`~vmex.core.step.StepControl`.

Convergence is declared when the *physical* residuals satisfy
``fsqr, fsqz, fsql <= ftolv`` simultaneously; the residual norms and the m=1
constraint rotation follow ``residue.f90`` (:mod:`vmex.core.residuals`).

Time-step control
~~~~~~~~~~~~~~~~~

The Garabedian-style update above is guarded by VMEC2000's ``DELT``
controller. Per step it computes

.. math::

   \tau_n = \min\left(\left|\ln\frac{\mathrm{fsq}_n}{\mathrm{fsq}_{n-1}}\right|,\; 0.15\right),

and maintains a moving average :math:`\overline{\tau}` over ``ndamp`` steps.
The damping factor is

.. math::

   \Delta\tau = \frac{\Delta t\,\overline{\tau}}{2}, \qquad
   b_1 = 1-\Delta\tau, \qquad
   \mathrm{fac} = \frac{1}{1+\Delta\tau},

giving the update :math:`\dot{x} \leftarrow \mathrm{fac}\,(b_1\dot{x} +
\Delta t\,F)`, :math:`x \leftarrow x + \Delta t\,\dot{x}` with :math:`F` the
preconditioned residual vector (``gc``). The controller tracks the minimum of
the preconditioned residual (``res0``) and the physical residual (``res1``);
growth triggers the restart rules below.

Restart control (``restart.f``)
-------------------------------

The loop keeps a checkpoint of the best state and applies VMEC2000's exact
back-off rules (:func:`~vmex.core.step.restart_decision` classifies the
step as ``STEP_OK``/``RESTART_JACOBIAN``/``RESTART_GROWTH``;
:func:`~vmex.core.step.apply_restart` restores the checkpoint, zeroes
the velocity and rescales ``delt``):

- **bad Jacobian** (``irst = 2``): restore the checkpoint, zero the velocity,
  ``delt *= 0.90``; on the first bad Jacobian the axis guess is recomputed
  (``guess_axis``), and ``delt`` is reset at ``ijacob = 25, 50`` with a hard
  stop at 75 (``jac75_flag``).  VMEX then offers a bounded driver-level
  recovery (two attempts by default): restart the best finite checkpoint with
  zero velocity and half the preceding initial ``DELT`` (capped at 0.5).
  This is a continuation, not a fresh ``profil3d`` initialization: the
  first-pass ``LMOVE_AXIS`` transfer is disabled on the driver-level retry, so
  a still-large force cannot replace the checkpoint with a cold axis-derived
  state.
  The force equations and stopping tolerance do not change.  Set
  ``jacobian_retries=0`` (Python) or ``--jacobian-retries 0`` (CLI) for the
  exact VMEC2000 fatal-stop policy.  Free-boundary recovery rebuilds the
  axis-current filament and all resolution/geometry-dependent NESTOR
  structures before continuing;
- **residual blow-up** (``irst = 3``): if after more than 10 steps the
  residual exceeds :math:`10^4\times` the checkpoint value, restore and
  ``delt /= 1.03``.

Constraint strength (``tcon``)
------------------------------

The spectral-condensation constraint force of
:doc:`spectral-representation` is scaled per surface by

.. math::

   \mathrm{tcon}(j) = \min\!\left(\left|\frac{a_{rd}}{a_{r,\mathrm{norm}}}\right|,
   \left|\frac{a_{zd}}{a_{z,\mathrm{norm}}}\right|\right)\cdot
   \mathrm{tcon}_0\text{-scaled}\cdot(32\,h_s)^2,
   \qquad \mathrm{tcon}(ns) = \tfrac{1}{2}\,\mathrm{tcon}(ns-1)

(``bcovar.f``). Implemented in :mod:`vmex.core.forces` (constraint force)
and :mod:`vmex.core.fields` (``tcon``). Preconditioner matrices, force
norms, and ``tcon`` are recomputed every ``ns4 = 25`` iterations and reused
in between — this cadence is parity-critical and is mirrored exactly.

Two execution lanes, one physics
--------------------------------

:mod:`vmex.core.solver` exposes the same jitted iteration through two
lanes (selected by ``vmex --mode cli|jit``): the default **CLI lane**, a
Python ``while`` loop around a jitted *N-iteration block* kernel with host
residual checks between blocks (exact-``ftol`` early exit, live
VMEC2000-format printing every ``NSTEP`` iterations, buffer donation, zero
autodiff bookkeeping), and the **JIT lane**, a single ``lax.while_loop``
over the same physics — fully traceable, the forward solver inside the
differentiable API. A regression test asserts per-block state agreement
between the lanes to machine precision. Which device (CPU or GPU) a lane
runs on is decided by the measured placement policy of
:mod:`vmex.core.device` — see :doc:`/howto/run-on-gpu` and
:doc:`architecture`.
