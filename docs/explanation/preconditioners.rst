Preconditioners
===============

Two preconditioners keep VMEX iterations cheap: the always-on 1D radial
tridiagonal preconditioner ported from ``precondn.f``/``scalfor.f``, and an
optional matrix-free 2D block preconditioner (``precon2d.f`` analogue) that
takes inexact Newton steps on stiff decks — measured 2.5–11x fewer
iterations on the stiff benchmark cases (:doc:`/reference/performance`).
Both plug into the Richardson update of :doc:`iteration`.

1D radial preconditioner (``precondn.f``, ``scalfor.f``)
--------------------------------------------------------

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
----------------------------------------

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
:doc:`/reference/vmec2000-compatibility`.
