# Adjoint gradients and SOLVAX

VMEX differentiates converged fixed-boundary equilibria by applying the
implicit function theorem at the converged fixed point — one linear solve per
scalar objective, O(1) memory in the iteration count, no unrolling, no
finite-difference step size. All of that linear algebra runs on
[SOLVAX](https://pypi.org/project/solvax/), the linear/adjoint solver layer
factored out of this code base; this page states the formulation and the six
SOLVAX solve classes exactly as the code uses them.

## The implicit function theorem on the fixed point

The equilibrium is the root of the force residual $F(x, p) = 0$ with $x$ the
spectral state and $p$ the parameters
({class}`~vmex.core.implicit.ImplicitParams`: boundary coefficients,
profiles, `phiedge`, `pres_scale`, `curtor`). If $\partial F/\partial x$ is
invertible at the root, the solution map $p \mapsto x^\star(p)$ is
differentiable with

$$
\frac{dx^\star}{dp}
= -\left(\frac{\partial F}{\partial x}\right)^{-1}
  \frac{\partial F}{\partial p},
$$

and for a scalar objective $\mathcal{J}(x^\star(p))$ with cotangent
$\bar{g} = \partial\mathcal{J}/\partial x$, the reverse-mode (adjoint) form
needs **one** linear solve regardless of the number of parameters:

$$
\left(\frac{\partial F}{\partial x}\right)^{\!\top} \lambda = \bar{g},
\qquad
\frac{d\mathcal{J}}{dp}
= -\lambda^{\!\top}\,\frac{\partial F}{\partial p}.
$$

{func}`~vmex.core.implicit.solve_implicit` wraps this in `jax.custom_vjp`:
the forward pass runs the fast CLI-lane host solver (`jax.pure_callback` —
multigrid staging, restarts, and adaptive time-step control stay invisible to
autodiff; only the fixed point defines the derivative); the backward pass
solves the adjoint system matrix-free
({func}`~vmex.core.implicit.adjoint_matvec`): one `jax.vjp` linearization of
the residual function ({func}`~vmex.core.implicit.residual_fn`) is reused as
the transposed operator, and one more VJP contracts $\lambda$ against
$\partial F/\partial p$.

The residual is the **self-consistently 1D-preconditioned force** `gc` of a
single fresh {func}`~vmex.core.solver.evaluate_forces` pass:
$F = M(x,p)\,f(x,p)$ with $f$ the raw spectral force and $M$ the invertible
1D-preconditioner map. At the root $dF = M\,df + dM\,f = M\,df$ up to
$O(\mathrm{ftol})$, so the implicit gradients equal those of the raw force to
solver accuracy — while the adjoint Krylov solve inherits VMEC's own
preconditioning for free: near equilibrium $\partial F/\partial x$ is close
to the identity, so it converges in a handful of iterations.

Why the gradient is cheap: reverse-mode through an *unrolled* iteration would
store every iterate (memory linear in the iteration count) and backpropagate
through thousands of steps. The implicit adjoint touches only the converged
state: its cost is a fixed handful of residual evaluations (one linearization
plus the Krylov matvecs) and its memory is O(1) in the iteration count —
independent of how many Richardson steps, restarts, or multigrid stages the
forward solve needed. Multigrid stages act purely as an initializer and are
stop-gradient by construction.

## The six SOLVAX solve classes

Every linear solve in the gradient stack goes through SOLVAX. The complete
inventory, with the call site each class serves:

1. `solvax.gmres` / `solvax.gcrot` solve the implicit-function-theorem
   systems in `vmex/core/implicit.py`: adjoint $(dF/dz)^T \lambda = b$
   (warm-started GMRES; GCROT(m,k) with subspace recycling) and tangent
   $(dF/dz)\, dz = -(dF/dp)\, t$ via the multi-RHS drivers
   {func}`~vmex.core.implicit.implicit_state_tangent_multi_rhs` /
   {func}`~vmex.core.implicit.implicit_state_pullback_multi_rhs`;
   every solve returns a {class}`~vmex.core.implicit.LinearResponseReport`
   (residual_norm, tolerance, iterations, converged).
2. `solvax.block_thomas_factor/solve` power the amortized Jacobian path in
   `vmex/core/optimize.py`: the raw scalxc-scaled residual Jacobian is
   block-tridiagonal in radius; assembled with 3-colored jvp probes,
   factored once, backsolved per boundary dof, then one warm-started GMRES
   pass per column certifies `cfg.adjoint_tol`.
3. `solvax.tridiagonal_solve(_checked)` performs the per-mode radial 1D
   preconditioner solves (`vmex/core/preconditioner.py`; the `precondn.f` /
   `scalfor.f` analogue — see {doc}`preconditioners`).
4. `solvax.gmres` solves the matrix-free 2D block-preconditioner Newton
   direction (`vmex/core/preconditioner_2d.py`; matvec
   `v -> jvp(g, state, v)`; the `precon2d.f` analogue).
5. `solvax.chunk_map` / `auto_chunk_size` bound memory for Jacobian columns
   and multi-RHS batches (`vmex/core/optimize.py`).
6. `solvax.SpluFactorization` owns the pivoted sparse factorization used to
   eliminate the radial bulk in the experimental free-boundary Schur
   transpose (`vmex/core/freeboundary_implicit.py`). VMEX supplies the
   physics-specific radial blocks and NESTOR edge response; SOLVAX owns the
   reusable factorization and transposed solves.

The mirror lane keeps its own adjoint solver (`vmex/mirror/implicit.py`).

## Certificates

No solve is trusted silently. Each adjoint/tangent solve returns a
{class}`~vmex.core.implicit.LinearResponseReport` with the achieved residual
norm, the requested tolerance, the iteration count, and a converged flag; the
block-Thomas Jacobian path certifies every column with one warm-started GMRES
pass against the preconditioned system to the same `adjoint_tol` as the
per-column path. SOLVAX's own GCROT result additionally carries a
`recycle_drift` monitor — how far the operator has drifted since the recycle
pair was built — but VMEX neither reads nor surfaces it, so those four fields
are the whole certificate.

## Validating the gradients: the frozen path

The adjoint returns the derivative of the fixed point of the **frozen**
residual $F$: the preconditioner, the `tcon` constraint strength, the
converged m=1 Z-force branch, and the dof mask are captured once at the base
parameters and held fixed. Validation must respect that. Equilibrium outputs
fall into two classes:

- **Smooth bulk integrals** — the magnetic energy `wb`, the aspect ratio,
  the volume. A naive central finite difference through the full host solver
  (re-converging independently at $p\pm h$) matches `jax.grad` to
  rtol <= 1e-6; the solver's internal path averages out of a bulk integral.
- **Solver-sensitive metrics** — `iota` (built from the current-constrained
  `chips` at `ncurr = 1`), the mirror ratio, the magnetic well, the
  Boozer/QI residual: these read the converged state directly and locally.
  A naive re-solve at $p\pm h$ lets the convergence logic re-form slightly
  differently on each side — an $O(1)$ perturbation of the discrete *path*,
  not of the *fixed point* — and it can swamp, even sign-flip, the finite
  difference. On `li383_low_res`,
  $d(\iota_{\mathrm{edge}})/d(\mathrm{RBC}(-1,1))$ is $-0.773$ from the
  adjoint but $+0.045$ from a naive central FD.

The naive FD is therefore not a valid reference for solver-sensitive metrics —
the disagreement is a property of the finite-difference probe, not an error
in the adjoint. The correct check reuses the *same* frozen residual the
adjoint differentiates: {func}`~vmex.core.implicit.frozen_path_directional_fd`
takes a directional step $p\pm h$, Newton-solves the frozen $F$ to its
perturbed root, and finite-differences that. It reproduces the adjoint to
solver accuracy for `iota`, mirror, well, and QI alike, and it is the
reference used in `tests/test_implicit_grad.py`.

## Forward mode for least-squares Jacobians

The adjoint is the right tool for one scalar objective and many parameters.
Vector residuals need the full Jacobian over all boundary dofs, which
`jac_solver="auto"` computes in **forward** mode: per dof tangent $t_j$, the
state response is

$$
dz_j = -\left(\frac{\partial F}{\partial z}\right)^{-1}
       \frac{\partial F}{\partial p}\, t_j,
$$

one linear solve per dof. Rather than an independent GMRES per column, the
vector-residual path (`jac_solver="block"`, SOLVAX class 2 above) exploits a
structural fact: in the **raw** force formulation the radial coupling of
$\partial F/\partial z$ is exactly nearest-neighbor, so the operator is
*exactly* block-tridiagonal — `ns` dense $(3\,mn \times 3\,mn)$ blocks.
(The preconditioned formulation used by the adjoint is dense in radius,
because the 1D preconditioner's inverse is.) Measured: the warm Jacobian
phase of the benchmark optimization step drops from 20.35 s to 0.61 s (33x;
`docs/_static/figures/gradient_stack_speedup.png`, reproduced by
`docs/_static/figures/sources/make_optimization_docs_figures.py`). The same
per-dof responses $dz_j$ double as a first-order perturbation warm start for
the optimizer's next trial solves — the DESC-style `eq.perturb` pattern —
measured 3.7x fewer total forward iterations over 20 trials (23,685 to
6,364). How these plug into an optimization campaign is
{doc}`/howto/optimize-a-boundary`.

```{figure} /_static/figures/gradient_stack_speedup.png
:alt: measured before/after of the three gradient-stack optimizations
:width: 100%

Measured gradient-stack speedups on the nfp2 minimal-seed deck (Jacobian
phase and trial iterations) and the full QA campaign (right); 2026-07-12,
CPU. Regenerate with
`python docs/_static/figures/sources/make_optimization_docs_figures.py`.
```

## Free-boundary root

The free-boundary path differentiates the converged coupled VMEX--NESTOR root,
including direct coil-shape/current parameters, while keeping host iterations
off the AD tape. The default whole-state transpose is exact but cold-compile
and memory limited. An advanced boundary-Schur transpose, its full-residual
certificate, and its current performance limits are documented in
{doc}`nestor-vacuum`. The supported scope of every AD path is one row of
{doc}`/reference/capabilities`.
