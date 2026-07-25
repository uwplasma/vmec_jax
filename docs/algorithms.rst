Algorithms
==========

This page documents the numerical algorithms of the ``vmex`` solver and
how they map onto their VMEC2000 counterparts. The module-by-module map is in
:doc:`architecture`; the underlying equations are derived in
:doc:`equations`.

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
:math:`\sqrt{s}` factor removed (``scalxc``), and R/Z and :math:`\lambda`
evolution starts from the m-dependent ``jmin2``/``jlam`` radial indices
(``vmec_params.f``). These conventions are implemented in
:mod:`vmex.core.geometry` and :mod:`vmex.core.setup`.

Angular grids and Fourier transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uniform tensor-product grids in :math:`\theta` and :math:`\zeta` (one field
period) with VMEC's symmetry-reduced theta extent. VMEC does **not** use a
plain FFT: the analysis transform (``tomnsps``) is a two-stage weighted DFT
with endpoint half-weights and ``mscale/nscale`` normalization baked into
precomputed trig tables (``fixaray.f``):

.. math::

   \tilde{f}^{c}_{m}(\zeta_k) = \sum_i f(\theta_i,\zeta_k)\,\mathrm{cosmui}_{i,m},
   \qquad
   c_{m,n} = \sum_k \tilde{f}^{c}_m(\zeta_k)\,\mathrm{cosnv}_{k,n},

and analogously for the sine tables. Derivative terms use ``cosnvn/sinnvn``,
which already include the field-period scaling :math:`n\,\mathrm{NFP}`. The
tables are built in :mod:`vmex.core.fourier` and the transforms
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

The iteration loop
------------------

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

Restart control (``restart.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

1D radial preconditioner (``precondn.f``, ``scalfor.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 1D preconditioner approximates the diagonal (in ``(m,n)``) of the
linearized radial force operator: for each spectral column the R/Z force is
replaced by the solution of a radial tridiagonal system

.. math::

   \bigl[\,b_x(s),\; d_x(s),\; a_x(s)\,\bigr]\,X = F_{mn}(s), \qquad
   d_x(s) = -\bigl(a_{xd} + b_{xd}\,m^2 + c_x\,(n\,\mathrm{NFP})^2\bigr),

whose coefficients are flux-surface integrals over the half mesh
(``precondn.f``, :func:`~vmex.core.preconditioner.precondn`) of
:math:`p_\tau = -4\,r_{12}^2\,\mathrm{bsq}\,w/\sqrt{g}`-type quantities: the
poloidal-derivative couplings give :math:`a_x`, the radial-derivative
couplings :math:`b_x`, and
:math:`c_x = \langle \tfrac14 p_{\mathrm{factor}} (B^v)^2 \sqrt{g}\rangle`
the toroidal couplings, each with even-m and odd-m columns (the odd column
carries the internal :math:`\sqrt{s}` scalings). Assembly of the per-mode
system with the :math:`m^2` and :math:`(n\,\mathrm{NFP})^2` weights, the
``edge_pedestal = 0.05`` and ZC(0,0)(ns) ``fac = 0.25`` stabilizations of
``scalfor.f``, and the ``jmin`` axis-row rules is
:func:`~vmex.core.preconditioner.scalfor_matrices`; the application is
:func:`~vmex.core.preconditioner.scalfor`. The solve is a Thomas
algorithm vectorized over all spectral columns simultaneously
(:func:`vmex.core.preconditioner.tridiagonal_solve`, a thin arg-order
adapter over ``solvax.tridiagonal_solve`` — the shared SOLVAX linear-solver
package). Production application uses ``solvax.tridiagonal_solve_checked``:
the unregularized Thomas pivots must pass VMEC2000's
``abs(pivot) > 1e-8*abs(diagonal)`` condition and a backward-residual check.
Rejected columns receive an identity preconditioner action and a typed
diagnostic instead of an amplified finite or NaN/Inf update. :math:`\lambda`
uses the diagonal ``faclam`` factors from
``lamcal.f90`` (:func:`~vmex.core.preconditioner.lamcal`):

.. math::

   \mathrm{faclam} \propto
   \frac{\sqrt{s}^{\,\min(m^2/16^2,\,8)}}
        {b_\lambda\,(n\,\mathrm{NFP})^2 \pm 2mn\,\mathrm{NFP}\,d_\lambda
         + c_\lambda\,m^2},

with :math:`b_\lambda = \langle g_{uu}/\sqrt{g}\rangle`,
:math:`c_\lambda = \langle g_{vv}/\sqrt{g}\rangle`,
:math:`d_\lambda = \langle g_{uv}/\sqrt{g}\rangle` (the :math:`\sqrt{s}`
damping only bites for :math:`m > 16`).

Preconditioner matrices, force norms, and the constraint multiplier ``tcon``
are recomputed every ``ns4 = 25`` iterations and reused in between — this
cadence is parity-critical and is mirrored exactly.

2D block preconditioner (``precon2d.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For stiff cases (high beta, high aspect ratio, high mode number) VMEC2000
optionally switches to its 2D preconditioner: a **Newton step** on the
1D-preconditioned force. Let :math:`g(\mathbf{x})` be the 1D-preconditioned
spectral force map (with the ``ns4`` cache frozen, so the 1D operator is a
fixed linear map during the Newton solve). The update direction solves

.. math::

   J\,\delta = -g(\mathbf{x}), \qquad
   J = \frac{\partial g}{\partial \mathbf{x}}
   \quad\text{(block-tridiagonal in radius)}.

Because the invertible 1D operator :math:`M_{\mathrm{1D}}^{-1}` is baked
into both :math:`J` and the right-hand side, it cancels exactly:
:math:`\delta` is the same full Newton step
:math:`-(\partial F/\partial\mathbf{x})^{-1} F` on the raw force — the 1D
preconditioner only conditions the linear solve (near equilibrium
:math:`M_{\mathrm{1D}}` approximates
:math:`\partial F/\partial \mathbf{x}`, so :math:`J \approx I`).

VMEC2000 (``Sources/Hessian/precon2d.f``) builds :math:`J` explicitly by
finite-difference "jogs" of every spectral column and LU-factors it with
BCYCLIC. In :mod:`vmex.core.preconditioner_2d` the force map is
traceable, so :math:`J v` is an **exact Hessian-vector product** from one
``jax.jvp`` (:func:`~vmex.core.preconditioner_2d.flat_operator`) — no
jogs, no assembled blocks — and the system is solved with matrix-free
restarted GMRES from SOLVAX (``solvax.gmres``) in
:func:`~vmex.core.preconditioner_2d.newton_direction`. A loose GMRES
tolerance yields an inexact Newton step; peak memory stays at one force
graph. Activation mirrors the main ``evolve.f`` gates
(:class:`~vmex.core.preconditioner_2d.Prec2DConfig`): finest grid only,
``iter2 >= 10``, and ``fsqr + fsqz + fsql < prec2d_threshold``; the wiring in
:mod:`vmex.core.solver` swaps the Newton direction for the 1D force
direction under a ``lax.cond``, leaving the default 1D-only path untouched.
It does **not** reproduce VMEC2000's distinct CG/GMRESR/TFQMR evolution
algorithms or its ``PRE_NITER`` budget mutation; see
:doc:`vmec2000_compatibility`.

Spectral condensation (``alias.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constraint force that drives spectral condensation is assembled from

.. math::

   z_{\mathrm{temp}} = (r_{\mathrm{con}} - r_{\mathrm{con},0})\,r_{\theta,0}
                     + (z_{\mathrm{con}} - z_{\mathrm{con},0})\,z_{\theta,0},

spectrally filtered to :math:`m \in [1, \mathrm{mpol}-2]` with the
``faccon(m)`` weights, and scaled per surface by

.. math::

   \mathrm{tcon}(j) = \min\!\left(\left|\frac{a_{rd}}{a_{r,\mathrm{norm}}}\right|,
   \left|\frac{a_{zd}}{a_{z,\mathrm{norm}}}\right|\right)\cdot
   \mathrm{tcon}_0\text{-scaled}\cdot(32\,h_s)^2,
   \qquad \mathrm{tcon}(ns) = \tfrac{1}{2}\,\mathrm{tcon}(ns-1)

(``bcovar.f``). Implemented in :mod:`vmex.core.forces` (constraint force)
and :mod:`vmex.core.fields` (``tcon``).

Two execution lanes, one physics
--------------------------------

:mod:`vmex.core.solver` exposes the same jitted iteration through two
lanes (selected by ``vmex --mode cli|jit``):

- **CLI lane** (default): a Python ``while`` loop around a jitted
  *N-iteration block* kernel, with residuals checked on the host between
  blocks. This gives exact-``ftol`` early exit, live VMEC2000-format printing
  every ``NSTEP`` iterations, and buffer donation, with zero autodiff
  bookkeeping.
- **JIT lane**: a single ``lax.while_loop`` over the same physics — fully
  traceable, used as the forward solver inside the differentiable API.

A regression test asserts per-block state agreement between the lanes to
machine precision. Which device (CPU or GPU) a lane runs on is decided by
the measured placement policy of :mod:`vmex.core.device` — see
:ref:`architecture:Device policy (CPU/GPU)`.

Multigrid and hot restart (``runvmec.f``, ``interp.f``)
-------------------------------------------------------

:func:`vmex.core.multigrid.solve_multigrid` runs the ``NS_ARRAY``
ladder: each stage solves at its ``ns`` with its own
``FTOL_ARRAY``/``NITER_ARRAY`` entry, and the converged coefficients are
interpolated in :math:`\sqrt{s}`-internal form to the next finer grid
(``interp.f``): scale by ``scalxc``, extrapolate odd-m modes to the axis on
the scaled array, interpolate linearly in :math:`s`, unscale, and zero the
odd-m axis row (:func:`~vmex.core.multigrid.interpolate_coefficients` /
:func:`~vmex.core.multigrid.interpolate_state`; the equations are in
:doc:`equations`). Each distinct stage structure may compile once; repeated
ladders with the same structures reuse those executables. A single
maximum-resolution masked executable is future work, not current behavior.
The same interpolation seam provides hot restart
(:func:`vmex.core.solver.hot_restart_state`): a previous solution (e.g.
the previous point of a parameter scan) can seed the solve directly, at the
same or a different radial resolution.

The transfer includes VMEC2000's non-geometric module state.  In particular,
``initialize_radial.f`` resets ``fsq``, ``iter1``, ``iter2``, ``ijacob``,
and the time-step controller, but it does **not** reset
``fsqr/fsqz/fsql``.  VMEX therefore passes the previous stage's three
invariant residuals into the first force evaluation on the next grid.  This
matters in free boundary: ``residue.f90`` uses the retained
``fsqr + fsqz`` to decide whether the carried edge-force row belongs in the
first fine-grid norm.  Axis re-guess transfers and bounded JAC75 retries
retain the same residual state instead of silently becoming cold starts.

Free boundary (NESTOR)
----------------------

For ``LFREEB = T`` decks, :mod:`vmex.core.vacuum` implements Merkel's
Green's-function method (NESTOR, J. Comp. Phys. 66, 83 (1986)). In the
vacuum region the field is curl-free, so it is written as

.. math::

   \mathbf{B}_{\mathrm{vac}} = \mathbf{B}_{\mathrm{ext}} + \nabla\Phi,
   \qquad \nabla^2 \Phi = 0,

with :math:`\mathbf{B}_{\mathrm{ext}}` the field of the external coils
(mgrid or Biot–Savart) plus the net-toroidal-current filament, and the
plasma boundary acting as a flux surface:

.. math::

   \mathbf{n}\cdot(\mathbf{B}_{\mathrm{ext}} + \nabla\Phi) = 0
   \quad \text{on } \partial\Omega.

Green's second identity turns this exterior Neumann problem into a boundary
integral equation for the surface potential,

.. math::

   \frac{\Phi(\mathbf{x}')}{2}
   = \oint_{\partial\Omega} \Bigl[
     \Phi(\mathbf{x})\,\mathbf{n}\cdot\nabla G(\mathbf{x},\mathbf{x}')
     + G(\mathbf{x},\mathbf{x}')\,
       \mathbf{n}\cdot\mathbf{B}_{\mathrm{ext}}(\mathbf{x})
     \Bigr]\, dS, \qquad
   G = \frac{1}{4\pi\,|\mathbf{x}-\mathbf{x}'|},

which, after expanding :math:`\Phi` in Fourier harmonics
:math:`\sin(mu - nv)/\cos(mu - nv)` on the boundary, becomes a dense
``mnpd2 x mnpd2`` linear system for the potential coefficients ``potvac``.
The :math:`|\mathbf{x}-\mathbf{x}'| \to 0` singularity of :math:`G` is split
off and integrated analytically (``analyt.f``, the ``cmns`` coefficient
tables); the regular remainder is tabulated on the angular grid (``greenf`` /
``fourp``). Implementation: geometry-independent tables in
:func:`~vmex.core.vacuum.vacuum_basis`, the jitted full/incremental
solves in :func:`~vmex.core.vacuum.make_vacuum_solver`, and the surface
field :math:`B_u = \mathrm{bexu} + \partial_u\Phi` (etc.) with
:math:`\mathrm{bsqvac} = |B_{\mathrm{vac}}|^2/2` in
:func:`~vmex.core.vacuum.vacuum_channels`.

:func:`vmex.core.freeboundary.solve_free_boundary` drives the coupling
with the VMEC2000 cadence (``funct3d.f``):

- the vacuum solve activates once :math:`\mathrm{fsqr}+\mathrm{fsqz} \le 10^{-3}`;
- a **full** NESTOR solve runs when ``mod(iter2 - iter1, nvacskip) == 0``,
  with cheaper incremental updates in between, and the cadence adapts as

  .. math::

     \mathrm{nvacskip} \leftarrow \max\!\left(\mathrm{nvskip}_0,\;
     \frac{1}{\max(0.1,\; 10^{11}\,(\mathrm{fsqr}+\mathrm{fsqz}))}\right);

- the vacuum pressure enters the edge force through
  ``rbsq = (bsqvac + presf_ns) * R(edge) / hs`` at ``js = ns``, and the
  constraint reference surfaces ``rcon0, zcon0`` ramp by 0.9 per iteration.

:func:`vmex.core.multigrid.solve_free_boundary_multigrid` implements
``runvmec.f``'s radial ladder.  Increasing grids interpolate ``xstore`` using
the same odd-m :math:`\sqrt{s}` scaling as fixed boundary; equal grids rerun
the current state and the ladder stops at the first decreasing entry.
``ivac``, adaptive ``nvacskip``, the exact ``rbsq`` edge product, and the
three invariant residual channels are carried.  Because the free-boundary
block is guarded by ``iter2 > 1``, a new stage uses that carried edge product
on iteration 1 and performs its first full update on iteration 2.  The
resolution-specific NESTOR basis, Green-function program, axis-current
filament program, cached potential matrix, and traced cadence loop are selected
or rebuilt for the new stage.  Vacuum activates only once across the ladder.

The forward NESTOR solver consumes a :class:`~vmex.core.mgrid.MgridField`.
It may be loaded from an ``mgrid`` file (trilinear interpolation weighted by
``EXTCUR``) or built once with
:meth:`~vmex.core.mgrid.MgridField.from_cartesian_field`, which tabulates an
ESSOS/SIMSOPT Biot--Savart object or any ``xyz -> B`` callable.  The resulting
table and its current scale remain JAX-differentiable; tabulation itself does
not retain coil-geometry derivatives.  Direct, interpolation-free ESSOS coil
derivatives use the virtual-casing residual below. vmex carries no coil code.

Differentiable free boundary (virtual casing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NESTOR iteration above is a host-driven fixed point and is not
differentiated. For coil/current optimization,
:mod:`vmex.core.freeboundary_diff` instead expresses the free-boundary
condition as a smooth objective on a given boundary. At the plasma-vacuum
interface the total exterior field
:math:`\mathbf{B}_{\mathrm{out}} = \mathbf{B}_{\mathrm{coil}} +
\mathbf{B}_{\mathrm{plasma}}` must be tangent, and pressure balance holds:

.. math::

   \mathbf{B}_{\mathrm{out}}\cdot\mathbf{n} = 0, \qquad
   |\mathbf{B}_{\mathrm{in}}|^2 + 2\mu_0 p = |\mathbf{B}_{\mathrm{out}}|^2.

The plasma's own exterior field comes from the **virtual-casing principle**:
the field produced outside :math:`\partial\Omega` by the plasma currents
equals that of the surface current
:math:`\mathbf{K} = \mathbf{n}\times\mathbf{B}/\mu_0` on
:math:`\partial\Omega`, evaluated with an accurate on-surface singular
quadrature (reused from ``virtual_casing_jax``;
:func:`~vmex.core.freeboundary_diff.surface_field_data_from_wout`
adapts a converged boundary + field, and
:func:`~vmex.core.freeboundary_diff.plasma_field_on_boundary` evaluates
the integral). The key structural fact: for a *fixed* trial boundary,
:math:`\mathbf{B}_{\mathrm{plasma}}` on that boundary does not depend on the
coil degrees of freedom, so it is precomputed once and frozen. The residual
assembled by
:class:`~vmex.core.freeboundary_diff.FreeBoundaryDiffProblem` is then a
smooth JAX function of the external-field dofs alone (coil Fourier
coefficients/currents of a callable ESSOS coil field via
:func:`~vmex.core.freeboundary_diff.external_B_cartesian`, or
``extcur``), and its ``value_and_grad_bnormal`` helper returns gradients
validated against finite differences — no NESTOR adjoint is required.

Implicit differentiation
------------------------

Gradients of equilibrium properties are computed by implicit differentiation
of the converged fixed point (:mod:`vmex.core.implicit`). The equilibrium
is the root of the force residual :math:`F(x, p) = 0` with :math:`x` the
spectral state and :math:`p` the parameters
(:class:`~vmex.core.implicit.ImplicitParams`: boundary coefficients,
profiles, ``phiedge``, ``pres_scale``, ``curtor``). By the implicit function
theorem, if :math:`\partial F/\partial x` is invertible at the root, the
solution map :math:`p \mapsto x^\star(p)` is differentiable with

.. math::

   \frac{dx^\star}{dp}
   = -\left(\frac{\partial F}{\partial x}\right)^{-1}
     \frac{\partial F}{\partial p},

and for a scalar objective :math:`\mathcal{J}(x^\star(p))` with cotangent
:math:`\bar{g} = \partial\mathcal{J}/\partial x`, the reverse-mode (adjoint)
form needs **one** linear solve regardless of the number of parameters:

.. math::

   \left(\frac{\partial F}{\partial x}\right)^{\!\top} \lambda = \bar{g},
   \qquad
   \frac{d\mathcal{J}}{dp}
   = -\lambda^{\!\top}\,\frac{\partial F}{\partial p}.

:func:`~vmex.core.implicit.solve_implicit` wraps this in
``jax.custom_vjp``:

- **forward**: runs the fast CLI-lane host solver (``jax.pure_callback`` —
  multigrid staging, restarts, and adaptive time-step control stay invisible
  to autodiff; only the fixed point defines the derivative) and returns the
  converged :math:`x^\star`;
- **backward**: solves the adjoint system matrix-free with restarted SOLVAX
  GMRES (:func:`~vmex.core.implicit.adjoint_matvec`): one ``jax.vjp``
  linearization of the residual function
  (:func:`~vmex.core.implicit.residual_fn`) is reused as the transposed
  operator, and one more VJP contracts :math:`\lambda` against
  :math:`\partial F/\partial p`.

The residual is chosen as the **self-consistently 1D-preconditioned force**
``gc`` of a single fresh :func:`~vmex.core.solver.evaluate_forces` pass:
:math:`F = M(x,p)\,f(x,p)` with :math:`f` the raw spectral force and
:math:`M` the invertible 1D-preconditioner map. At the root
:math:`dF = M\,df + dM\,f = M\,df` up to :math:`O(\mathrm{ftol})`, so the
implicit gradients equal those of the raw force to solver accuracy — while
the adjoint GMRES inherits VMEC's own preconditioning for free (this is the
"preconditioned adjoint": near equilibrium :math:`\partial F/\partial x`
is close to the identity, so GMRES converges in a handful of Krylov
iterations).

Why the gradient is cheap: reverse-mode through an *unrolled* iteration would
store every iterate (memory linear in the iteration count) and backpropagate
through thousands of steps. The implicit adjoint touches only the converged
state: its cost is a fixed handful of residual evaluations (one
linearization plus the GMRES matvecs) and its memory is O(1) in the
iteration count — independent of how many Richardson steps, restarts, or
multigrid stages the forward solve needed. Multigrid stages act purely as an
initializer and are stop-gradient by construction. See :doc:`optimization`
for usage and the references (Skene & Burns 2026; jaxopt; DESC) in
:doc:`references`.

Gradient checking: solver-sensitive metrics and the frozen path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adjoint returns the derivative of the fixed point of the **frozen**
residual :math:`F`: the preconditioner, the ``tcon`` constraint strength, the
converged m=1 Z-force branch, and the dof mask are all captured once at the
base parameters and held fixed, not re-derived as :math:`p` moves. This is
exactly the object the implicit function theorem differentiates, and it has a
subtle consequence for how the gradient must be *validated*.

Equilibrium outputs fall into two classes:

- **Smooth bulk integrals** — the magnetic energy ``wb``, the aspect ratio,
  the volume: volume averages of the converged geometry. For these a naive
  central finite difference through the full host solver
  (re-converging independently at :math:`p\pm h`) already matches
  ``jax.grad`` to :math:`\mathrm{rtol}\le 10^{-6}`. The solver's internal
  path — how many Richardson steps, which restarts fired — averages out of a
  bulk integral.

- **Solver-sensitive metrics** — the rotational transform ``iota`` (built
  from the current-constrained ``chips`` at ``ncurr = 1``), the mirror ratio,
  the magnetic well, the Boozer/QI residual: these read the converged state
  *directly and locally*. A naive re-solve at :math:`p\pm h` lets the
  convergence logic (restart timing, the m=1 rotation, the last accepted
  step) re-form slightly differently on each side. That is an :math:`O(1)`
  perturbation of the discrete *path*, not of the *fixed point*, and it can
  swamp — even sign-flip — the finite difference. On ``li383_low_res``,
  :math:`d(\iota_{\mathrm{edge}})/d(\mathrm{RBC}(-1,1))` is :math:`-0.773`
  from the adjoint but :math:`+0.045` from a naive central FD: the two do not
  even agree in sign.

The naive FD is therefore **not a valid reference** for solver-sensitive
metrics — the disagreement is a property of the finite-difference probe, not
an error in the adjoint. The correct check reuses the *same* frozen residual
the adjoint differentiates:
:func:`~vmex.core.implicit.frozen_path_directional_fd` takes a directional
step :math:`p\pm h`, Newton-solves the frozen :math:`F` to its perturbed root
(rather than re-running the full adaptive solver from scratch), and finite-
differences that. It reproduces the adjoint to solver accuracy for ``iota``,
mirror, well, and QI alike, and it is the reference used in
``tests/test_implicit_grad.py``. Bulk integrals are validated against both the
naive and the frozen-path FD; solver-sensitive metrics against the frozen-path
FD only. In short: the adjoint linearizes the fixed point, so a correct
numerical check must also hold the path fixed.

Forward-mode Jacobians for least squares (block-tridiagonal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adjoint above is the right tool for *one* scalar objective and many
parameters. A least-squares optimizer needs the opposite object — the full
residual Jacobian over all boundary dofs — which is computed in **forward**
mode: per dof tangent :math:`t_j`, the state response is

.. math::

   dz_j = -\left(\frac{\partial F}{\partial z}\right)^{-1}
          \frac{\partial F}{\partial p}\, t_j ,

one linear solve per dof. Rather than running an independent GMRES per
column, the default path (``jac_solver="block"``) exploits a structural
fact: in the **raw** force formulation the radial coupling of
:math:`\partial F/\partial z` is exactly nearest-neighbor (the
finite-difference stencil in :math:`s`), so the operator is *exactly*
block-tridiagonal — ``ns`` dense :math:`(3\,mn \times 3\,mn)` blocks.
(The preconditioned formulation used by the adjoint is dense in radius,
because the 1D preconditioner's inverse is.) The blocks are assembled with
3-colored ``jax.jvp`` probes — a cost independent of the dof count —
factored once with SOLVAX's block-Thomas elimination (the BCYCLIC
analogue), and back-substituted for every dof right-hand side; a short
warm-started GMRES pass against the preconditioned system certifies each
column to the same tolerance as the per-column path. Measured: 33x on the
Jacobian phase of the benchmark optimization step (see
:doc:`optimization`). The same per-dof responses :math:`dz_j` double as a
first-order perturbation warm start for the optimizer's next trial solves —
the DESC-style ``eq.perturb`` pattern — making the linearization pay twice.
