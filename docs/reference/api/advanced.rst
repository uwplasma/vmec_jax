Advanced API
============

Everything below :doc:`basic`: the solver internals of :mod:`vmex.core` and
the open-field-line lane :mod:`vmex.mirror`, grouped as in
:doc:`/explanation/architecture`. Every docstring names the VMEC2000
counterpart it ports.

The package docstring of :mod:`vmex.core` is the module map this page follows:

.. automodule:: vmex.core
   :no-members:

Profiles
--------

.. automodule:: vmex.core.profiles
   :members:

Radial basis and axis regularity
--------------------------------

.. automodule:: vmex.core.radial_basis
   :members:

High-order reconstruction and force certificate
------------------------------------------------

.. automodule:: vmex.core.strong_force
   :members:

High-order correction transfer and preconditioner
-------------------------------------------------

.. automodule:: vmex.core.polish
   :members:

.. automodule:: vmex.core.polish_driver
   :members:

.. automodule:: vmex.core.polish_implicit
   :members:

Both :func:`vmex.solve` and :func:`vmex.solve_multigrid` accept
``polish_force_balance=False`` (unchanged behavior), ``True`` (required
correction), or ``"auto"``.  ``"auto"`` skips an already-certified state,
and additionally times one Gauss--Newton linear product before committing
to the solve: if the configured iteration limits could run past
``PolishConfig.auto_budget_seconds`` (``POLISH_BUDGET``,
``--polish-budget``, default 3600 s) it reports the measurement, returns
the equilibrium unpolished and uncertified, and names the knobs that
override the decision.  It never raises, because nothing was attempted.
``True`` never measures and never declines. The shorter
``polish`` keyword remains an alias on the single-grid call. A standard VMEC
solve never polishes unless the caller explicitly requests it. A standard
VMEC deck enables the same path with comment directives that VMEC2000 ignores::

   !@VMEX POLISH = AUTO
   !@VMEX POLISH_TOL = 1.0E-8
   !@VMEX POLISH_FAIL = ERROR
   !@VMEX POLISH_DEGREE = 5
   !@VMEX POLISH_MAX_ITER = 40
   !@VMEX POLISH_SPANS = 16
   !@VMEX POLISH_BUDGET = 3600

(the original single-flag spelling ``! VMEX: POLISH_FORCE_BALANCE = .TRUE.``
still parses).  Directives are execution metadata, owned by
:mod:`vmex.core.run_options` — they never become :class:`vmex.VmecInput`
fields, and ``VmecInput.from_file`` ignores them while preserving all physics.
Structured JSON carries the same keys in a reserved ``_vmex`` section that is
removed before schema validation.  :func:`vmex.solve_file` runs a deck the way
the CLI does — directives honored, ``wout_<case>.nc`` written — and explicit
Python keywords override the file::

   result = vmex.solve_file("input.case", polish="auto")

Precedence is exactly ``CLI option > Python keyword > file directive >
package default``; the package default is false, and the CLI prints which
layer a polish request came from.
``polish_fail`` selects the failure behavior: ``"error"`` raises,
``"fallback"`` returns the unpolished state, ``"warn"`` does the same with a
:class:`RuntimeWarning` — never a failure silently presented as polished. A successful result retains the ordinary ``state`` and exposes
the VMEC-grid projection as ``polished_state``; an
:class:`vmex.core.optimize.Equilibrium` requested with polishing uses that
projected state, while the continuous solution remains in
``native_equilibrium``.  CLI WOUT files instead sample
``native_equilibrium`` on a denser radial mesh
(:func:`vmex.core.polish_driver.polished_wout_ns`): on the solve mesh the
stable wout reconstruction cannot resolve the between-node correction, so a
solve-resolution export would silently discard most of the certified gain.

Validated scope
^^^^^^^^^^^^^^^

Every case VMEX ships as a *certified* polish is axisymmetric
(``NTOR = 0``).  No 3-D deck has yet passed the independent certificate, and
the limit measured so far is cost rather than a demonstrated impossibility:
on the W7-X standard configuration (``MPOL = NTOR = 10``, ``ns = 51``) one
Gauss--Newton iteration takes about 1.75 h on 36 CPU cores, so the iteration
count that produced the one substantial 3-D improvement — a 40% reduction in
independent force error on a ``MPOL = NTOR = 5`` QA deck, which still failed
the certificate — is a multi-day run at production resolution.  This is what
``POLISH = AUTO`` measures and declines on.  The runs, their budgets, and
their certificates are recorded in ``benchmarks/polish3d_tuning.md``.

The certificate's ``normalized_l2`` divides the force error by the local
force scale ``|JxB| + |grad p|``.  On a vacuum or near-vacuum deck both terms
vanish and the ratio saturates at its ceiling of 2 whatever the equilibrium
quality, so on such decks read the dimensional ``absolute_l2`` instead.

The public primal path solves the overdetermined physical collocation residual
with SOLVAX Gauss--Newton and accepts it only after independent force,
radial-refinement, and nestedness checks.  A successful result carries a
``polish_context`` for :func:`vmex.collocation_polish_tangent`,
:func:`vmex.collocation_polish_adjoint`, and
:func:`vmex.implicit_collocation_polished_state`.  These differentiate the
exact least-squares stationarity equation, including its nonzero-residual
Hessian term, without replaying nonlinear iterations.  The earlier square-root
diagnostics remain internal to :mod:`vmex.core.polish_implicit`.

A scalar objective differentiates through that converged state directly.  This
example minimizes relative field-strength variation on one interior surface::

   import jax
   import jax.numpy as jnp
   import vmex as vj

   native = result.polish_context.runtime.native

   def objective(value):
       polished = vj.implicit_collocation_polished_state(
           value, result.polish_context)
       theta = jnp.linspace(0, 2 * jnp.pi, 12, endpoint=False)
       zeta = jnp.linspace(0, 2 * jnp.pi, 6, endpoint=False)
       tt, zz = jnp.meshgrid(theta, zeta, indexing="ij")
       B = vj.evaluate_high_order_fields(polished, 0.7, tt, zz).B
       magnitude = jnp.linalg.norm(B, axis=-1)
       return jnp.var(magnitude / jnp.mean(magnitude))

   gradient = jax.grad(objective)(native)

For boundary objectives, :func:`vmex.evaluate_high_order_surface` returns a
one-field-period array view accepted by ESSOS, and
:func:`vmex.surface_field_data_from_high_order` converts the same analytic
geometry and edge field for ``virtual_casing_jax``.  Neither path writes a
``wout`` file or finite-differences a surface tangent.  Field-aligned
objectives use :func:`vmex.boozer_spectrum_high_order`, which sends continuous
geometry and field tables to BOOZ_XFORM_JAX without reconstructing a sampled
radial mesh.

Spectral representation and physics kernels
-------------------------------------------

.. automodule:: vmex.core.fourier
   :members:

.. automodule:: vmex.core.transforms
   :members:

.. automodule:: vmex.core.geometry
   :members:

.. automodule:: vmex.core.fields
   :members:

.. automodule:: vmex.core.forces
   :members:

.. automodule:: vmex.core.residuals
   :members:

Solver
------

.. automodule:: vmex.core.setup
   :members:

.. automodule:: vmex.core.preconditioner
   :members:

.. automodule:: vmex.core.preconditioner_2d
   :members:

.. automodule:: vmex.core.step
   :members:

.. automodule:: vmex.core.solver
   :members:

.. automodule:: vmex.core.multigrid
   :members:

.. automodule:: vmex.core.restart
   :members:

.. automodule:: vmex.core.device
   :members:

Free boundary
-------------

.. automodule:: vmex.core.vacuum
   :members:

.. automodule:: vmex.core.freeboundary
   :members:

.. automodule:: vmex.core.freeboundary_implicit
   :members:

.. automodule:: vmex.core.freeboundary_linear
   :members:

.. automodule:: vmex.core.virtual_casing
   :members:

.. automodule:: vmex.core.freeboundary_diff
   :no-members:

It re-exports the names above unchanged; it does not differentiate through a
moving-boundary NESTOR equilibrium solve.

.. automodule:: vmex.core.mgrid
   :members:

.. automodule:: vmex.core.extender
   :members:

Physics objectives
------------------

The objective catalog with usage snippets is :doc:`/reference/objectives`.
The wout-parity scalar targets (aspect ratio, volume, beta, elongation, iota)
that the objective modules share live in one place and are re-exported by
:mod:`vmex.core.optimize`:

.. automodule:: vmex.core.statephysics
   :members:

.. automodule:: vmex.core.omnigenity
   :members:

.. automodule:: vmex.core.bounce
   :members:

.. automodule:: vmex.core.qi
   :members:

.. automodule:: vmex.core.maxj
   :members:

.. automodule:: vmex.core.gammac
   :members:

.. automodule:: vmex.core.bootstrap
   :members:

.. automodule:: vmex.core.stability
   :members:

.. automodule:: vmex.core.turbulence
   :members:

Outputs
-------

.. automodule:: vmex.core.scaling
   :members:

.. automodule:: vmex.core.nyquist
   :members:

.. automodule:: vmex.core.postprocess
   :members:

.. automodule:: vmex.core.printing
   :members:

.. automodule:: vmex.core.plotting
   :members:

.. automodule:: vmex.core.boozer
   :members:

The differentiable route from a spectral state to a single-surface Boozer
transform: traceable ``wout``-convention mode tables that ``booz_xform_jax``
consumes, so ``jax.grad`` flows from boundary coefficients through to
downstream kinetic codes.

.. automodule:: vmex.core.boozer_tables
   :members:

Straight-axis mirrors
---------------------

.. automodule:: vmex.mirror
   :no-members:

Collocation bases, geometry, and force kernels first; then the spline
discretization and the solves built on them; then the exterior vacuum used by
the free-boundary lane, derivatives, gyrokinetic geometry, and MOUT output.

.. automodule:: vmex.mirror.analytic
   :members:

.. automodule:: vmex.mirror.basis
   :members:

.. automodule:: vmex.mirror.geometry
   :members:

.. automodule:: vmex.mirror.forces
   :members:

.. automodule:: vmex.mirror.splines
   :members:

.. automodule:: vmex.mirror.model
   :members:

.. automodule:: vmex.mirror.solver
   :members:

.. automodule:: vmex.mirror.exterior
   :members:

.. automodule:: vmex.mirror.free_boundary
   :members:

.. automodule:: vmex.mirror.implicit
   :members:

.. automodule:: vmex.mirror.turbulence
   :members:

.. automodule:: vmex.mirror.output
   :members:

.. automodule:: vmex.mirror.metrics
   :members:

Errors and CLI
--------------

.. automodule:: vmex.core.errors
   :members:

.. automodule:: vmex.core.cli
   :members:

``vmex --doctor`` collects and formats its installation report here.

.. automodule:: vmex.doctor
   :members:
