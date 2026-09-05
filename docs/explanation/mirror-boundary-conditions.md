# Mirror boundary conditions

This page is the boundary-condition contract for the mirror lanes in
`vmex.mirror`: the variational statement each topology solves, the natural
boundary terms of the mirror energy, the mechanism that makes each term
vanish, and the place in the code that enforces it. Every row of the term
table is pinned by a directional-derivative test in
`tests/mirror/test_boundary_conditions.py`, so the derivation and the
implementation cannot drift apart silently. Coordinates, the field
representation, and the cut semantics are introduced in
{doc}`/explanation/mirror-geometry`; the toroidal analog of this page is
{doc}`/explanation/variational-problem`.

## Model contract by topology

**Open fixed-cut mirror.** Radial label $s \in [0,1]$, poloidal angle
$\theta$, axial coordinate $\xi \in [-1,1]$. The lateral surface $s=1$ is a
fixed boundary and the axis $s=0$ is a regular coordinate line. The two cuts
$\xi = \pm 1$ carry prescribed geometry and prescribed normal magnetic flux:
field lines cross them, so they are neither periodic identifications nor
plasma-vacuum interfaces. The model makes no claim about sheaths, end-loss
kinetics, or sources; it is the finite-domain magnetostatic equilibrium
between two flux-carrying planes.

**Open free-lateral-boundary mirror.** The side wall is the plasma-vacuum
interface of ideal MHD: $\mathbf{B}\cdot\mathbf{n} = 0$ on both sides and
continuity of the isotropic total pressure $p + B^2/(2\mu_0)$
(Hirshman and Whitson 1983). The anisotropic extension of this lane replaces
the continuity condition by $p_\perp + B^2/(2\mu_0)$ (Cooper et al. 1992,
2009); the isotropic lane documented here is its $p_\perp \to p$ limit. The
end caps that close the exterior Green surface are a mathematical closure
only — they continue the through-flux as Neumann data and receive no
side-wall pressure-balance condition.

**Periodic stellarator-mirror hybrid.** All geometry and field variables are
periodic in the longitudinal coordinate. There are no cuts and no end-cap
conditions. Field-line closure and rationality are diagnostic and selection
criteria evaluated on solutions ({func}`~vmex.mirror.trace_closed_field_line`),
not boundary conditions imposed on the solve.

## The energy and its first variation

The isotropic lanes minimize the mass-conserving energy of the steepest
descent moment method (Hirshman and Whitson 1983, section II), written on
mirror coordinates $(s, \theta, \xi)$ with Jacobian $\sqrt{g}$:

$$
W = \int \frac{B^2}{2\mu_0}\,\sqrt{g}\;ds\,d\theta\,d\xi
  + \int_0^1 \frac{M(s)}{(\gamma - 1)\,V'(s)^{\gamma-1}}\,ds,
\qquad p(s) = \frac{M(s)}{V'(s)^{\gamma}},
$$

where $V'(s) = \oint \sqrt{g}\, d\theta\, d\xi$ and the mass profile $M(s)$
is held fixed, so pressure forces emerge from the variation instead of being
bolted on. The independent variables are the radius map, stored as the
regular scale $a$ with $r = \sqrt{s}\,a(s,\theta,\xi)$, and the stream
function $\lambda$ of the divergence-free field representation

$$
\sqrt{g}\,B^\theta = I'(s) - \partial_\xi \lambda, \qquad
\sqrt{g}\,B^\xi = \Psi'(s) + \partial_\theta \lambda, \qquad B^s = 0 .
$$

### Stream-function variation

Because $\lambda$ enters only through derivatives, integrating the first
variation by parts in $\theta$ and $\xi$ gives

$$
\delta_\lambda W
 = -\int \sqrt{g}\, J^s\, \delta\lambda \; ds\,d\theta\,d\xi
 \;-\; \frac{1}{\mu_0} \left[\, \oint B_\theta\, \delta\lambda \; d\theta\,ds \right]_{\xi=-1}^{\xi=+1},
\qquad
\mu_0 \sqrt{g}\, J^s = \partial_\theta B_\xi - \partial_\xi B_\theta ,
$$

with $B_\theta, B_\xi$ the covariant field components. Interior
stationarity is the vanishing radial current $J^s = 0$ — the mirror analog
of VMEC's $\lambda$ force. The $\theta$ boundary pair cancels exactly by
periodicity of the poloidal basis. The cut term is the work done by the
tangential field against a redistribution of the through-flux: prescribing
the normal-flux distribution at the cuts means $\delta\lambda = 0$ there,
which removes it. The periodic hybrid has no cut term at all because the
longitudinal basis is periodic.

### Geometry variation

Varying the mapping by a displacement $\boldsymbol{\xi}$ that respects the
constraints reproduces the classic form (Hirshman and Whitson 1983)

$$
\delta_{\boldsymbol{\xi}} W
 = -\int \mathbf{F}\cdot\boldsymbol{\xi}\,\sqrt{g}\;ds\,d\theta\,d\xi
 \;-\; \oint_{s=1} \Bigl(p + \tfrac{B^2}{2\mu_0}\Bigr)\,
   \boldsymbol{\xi}\cdot\mathbf{n}\; dA
 \;-\; \sum_{\xi=\pm1} \oint T^{\xi}{}_{r}\, \delta r \; d\theta\,ds ,
$$

with $\mathbf{F} = \mathbf{J}\times\mathbf{B} - \nabla p$ and, on the cut
planes, the Maxwell-plus-pressure stress component $T^{\xi}{}_{r}$ conjugate
to a radial displacement $\delta r$ of the cut ring. For the axisymmetric
lateral surface the discrete area measure is
$\boldsymbol{\xi}\cdot\mathbf{n}\,dA = \delta a \; r\, z_\xi \, d\theta\,d\xi$
at $s = 1$, which is exactly the virtual-work measure used by the
free-boundary map (`vmex.mirror.free_boundary._spline_boundary_work`).

### Why each boundary term vanishes

| Term | Vanishes because | Enforced by |
|---|---|---|
| Lateral $s{=}1$, $\bigl(p + B^2/2\mu_0\bigr)\,\boldsymbol{\xi}\cdot\mathbf{n}$ | fixed geometry (fixed lane) or interface balance (free lane) | LCFS row overwritten in {func}`~vmex.mirror.model.project_fixed_boundary_state`; LCFS coefficients excluded from the solve vector; free lane drives {func}`~vmex.mirror.forces.interface_residual` to zero |
| Cut $\xi{=}\pm1$ geometry, $T^{\xi}{}_{r}\,\delta r$ | fixed cut geometry | cut radius coefficients excluded from the solve vector (`_SplineStateVectorizer`) |
| Cut $\xi{=}\pm1$ flux, $B_\theta\,\delta\lambda/\mu_0$ | fixed normal-flux distribution | cut $\lambda$ coefficients excluded from the solve vector |
| $\theta$ boundary pair | periodicity | Fourier poloidal basis |
| Longitudinal pair (hybrid) | periodicity | periodic B-spline basis; all axial coefficients active, no cut masks |
| Axis $s{=}0$ | regularity: odd radius modes vanish as $\sqrt{s}$, axis $\lambda$ fixed by single-valued axial flux | `_regularize_axis_radius` and {func}`~vmex.mirror.geometry.regularize_axis_stream_function` |
| Gauge $\lambda \to \lambda + f(s)$ | $W$ depends on $\lambda$ only through $\partial_\theta\lambda,\ \partial_\xi\lambda$ | per-surface zero weighted mean (nodal lane); weighted pivot elimination (coefficient lane) |

Each row is turned into a test by evaluating the discrete directional
derivative of the energy along a variation supported on exactly one family:
constrained families must be null directions of the projected gradient,
gauge shifts must leave the energy unchanged to round-off, and the two
released natural terms (cut flux, lateral displacement) must converge to the
surface integrals above under refinement. The lateral test evaluates a
solved equilibrium, where $\mathbf{F} \approx 0$ makes the total derivative
equal the natural term.

## The exterior boundary-value problem

The free-lateral-boundary lane closes the plasma with a *Green surface*
$S = S_{\rm lat} \cup S_- \cup S_+$: the lateral LCFS plus two graded end
disks. $S$ is a topological sphere, and the vacuum domain is the unbounded
exterior $E = \mathbb{R}^3 \setminus \overline{\Omega}$. Write the vacuum field
as the applied coil field plus a correction potential,

$$
\mathbf{B}_{\rm vac} = \mathbf{B}_{\rm ext} + \nabla\Phi ,
\qquad \nabla^2\Phi = 0 \ \text{in } E,
\qquad \Phi = O(|\mathbf{x}|^{-1}) \ \text{as } |\mathbf{x}| \to \infty .
$$

With $\mathbf{n}$ the unit normal pointing out of the plasma into the vacuum,
the Neumann data $g = \mathbf{n}\cdot\nabla\Phi$ are

$$
g = -\,\mathbf{n}\cdot\mathbf{B}_{\rm ext} \quad\text{on } S_{\rm lat},
\qquad
g = -\,\mathbf{n}\cdot\mathbf{B}_{\rm ext}
    \;+\; \mathbf{n}\cdot\mathbf{B}_{\rm plasma} \quad\text{on } S_\pm .
$$

The lateral row is the interface condition $\mathbf{B}\cdot\mathbf{n} = 0$; the
cap rows continue the plasma's axial through-flux into free space, because the
caps are a mathematical closure, not a material interface, and carry no
pressure-balance condition.

**This problem is uniquely solvable as written.** The exterior Neumann problem
with decay at infinity has exactly one solution for arbitrary data: the
$O(1/r)$ monopole term absorbs the net flux, so there is neither a solvability
condition on $g$ nor an additive-constant gauge freedom to fix. (Both of those
belong to the *interior* Neumann problem, where $\oint_S g\,dA = 0$ is required
and the solution is unique only up to a constant.) `LaplaceNeumannResult`
accordingly reports `gauge_error` as an identical zero.

What *is* enforced is a different, physical consistency requirement. A magnetic
field is solenoidal, so the exact data satisfy $\oint_S g\,dA = 0$; a nonzero
discrete value would be a spurious magnetic monopole inside $S$. The discrete
lateral and cap data are built from different interpolants and do not cancel to
round-off on their own, so `_balance_neumann_on_caps` adds one constant to the
*cap* rows only — the physical lateral $\mathbf{B}\cdot\mathbf{n}$ data are
untouched, which `test_axisymmetric_neumann_balance_changes_only_artificial_caps`
checks. `compatibility_error` and `raw_compatibility_error` report the residual
net flux after and before that projection.

**Discretization.** $S$ is triangulated (lateral quadrilaterals split in two,
caps graded toward the rim) and the direct boundary-integral identity is
collocated at the mesh vertices. With $G(\mathbf{x},\mathbf{y}) =
1/(4\pi|\mathbf{x}-\mathbf{y}|)$, the equation solved at each collocation point
$\mathbf{x}\in S$ is

$$
\Phi(\mathbf{x})
 + \int_S G(\mathbf{x},\mathbf{y})\, g(\mathbf{y}) \; dA_{\mathbf y}
 - \int_S \bigl[\Phi(\mathbf{y}) - \Phi(\mathbf{x})\bigr]\,
   \frac{\partial G}{\partial n_{\mathbf y}}(\mathbf{x},\mathbf{y}) \; dA_{\mathbf y}
 \;=\; 0 .
$$

Subtracting $\Phi(\mathbf{x})$ inside the double-layer integral makes constants
an exact null direction of the operator and removes the need to assume the
smooth-surface $1/2$ jump coefficient, which does not hold at the cap rims
where the surface has an edge. Both layer integrals are evaluated with Duffy's
vertex transform: each incident triangle is rotated so the collocation vertex is
Duffy's singular vertex, and the $1/r$ and $1/r^2$ kernels become bounded
integrands on the unit square under tensor Gauss-Legendre quadrature. The
resulting dense system in the symmetry-reduced unknowns is formed by
`jax.jacfwd` of the residual and solved directly, so the whole exterior response
is differentiable.

**Relation to NESTOR.** The physical problem — an ideal-MHD plasma-vacuum
interface closed by an exterior Neumann problem for a scalar potential — is the
one solved for stellarators by Merkel and by Hirshman, van Rij and Merkel
(references below). The numerics here are not NESTOR's: NESTOR works on a
toroidal surface in a Fourier basis, with an analytic singularity subtraction
and a net-toroidal-current filament folded into $\mathbf{B}_{\rm ext}$, while
this lane uses a triangulated topologically spherical surface with mathematical
end caps, Duffy quadrature and vertex collocation. The missing filament is
exactly why a net axial plasma current is inadmissible here: a single-valued
potential on a simply connected exterior carries no azimuthal field, so
`solve_free_boundary` and its relatives reject `current_derivative != 0`
({func}`~vmex.mirror.free_boundary.reject_net_axial_current`).

## Axis regularity audit

A single-valued smooth scalar has poloidal Fourier coefficients that behave
as $\rho^{|m|}$ times a smooth even function of $\rho = \sqrt{s}$ near a
polar axis; high-order polar discretizations build this structure into the
basis (arXiv:2601.17841). The stream-function interpolation implements the
full mode-dependent rule: `_interpolate_stream_function` removes
$\rho^{|m|}$, interpolates the smooth remainder linearly in $s$, and
restores the factor at quadrature points.

The radius scale $a$ is not a scalar field — it is the polar chart of the
mapping, parameterized by the geometric angle
$(x, y) = (r\cos\theta, r\sin\theta)$. Two facts follow, both demonstrated
numerically in `tests/mirror/test_boundary_conditions.py`:

1. **The odd/even parity rule is the correct per-mode factorization.** Odd
   modes of $a$ translate the limiting cross-section and must vanish as
   $\sqrt{s}$; even modes describe the centered limiting section and stay
   finite. For every admissible state, the remainder after removing the odd
   $\sqrt{s}$ factor is smooth in $s$, so the two-point Gauss radial
   interpolation reaches its $O(\Delta s^2)$ design order. Extracting the
   full $\rho^{|m|}$ from higher even modes changes constants, not order.
2. **Full per-mode $\rho^{|m|}$ factors on $a$ are inadmissible.** A
   supported state with finite axis ellipticity has $a_2(s{=}0) \neq 0$;
   dividing that mode by $s$ manufactures a $1/s$ singularity and destroys
   the interpolation near the axis. The genuine smoothness condition of the
   geometric-angle chart couples adjacent radius modes
   ($r_{m-1} + r_{m+1} \sim \rho^{|m|}\times$ even) rather than bounding
   each mode separately, and satisfying it exactly is a chart (poloidal
   angle) choice, not a per-mode weight.

The stored state therefore stays unchanged: the audit tests demonstrate the
current representation meets its design order for the supported mode
families, and that the only stronger per-mode rule would break supported
states. The residual chart question belongs to the rotating-ellipse
promotion work tracked in {doc}`/explanation/mirror-geometry`.

## Solver algebra and SOLVAX

The mirror lanes orchestrate their nonlinear solves on the host: bounded
L-BFGS-B globalization, damped Newton-GMRES polish with exact JAX Hessian
products, the separable tensor preconditioner, and a dense trust-region
rescue capped at 2048 unknowns (`vmex.mirror.solver`). SOLVAX 0.20.0 was
compared against this lane condition by condition:

- **Same equations and convergence contract** — not met. The mirror
  contract requires box bounds on normalized radius coefficients, an
  energy-merit line search that rejects sign-changed Jacobians with an
  infinite objective, and a damping schedule tied to step acceptance.
  `solvax.newton_krylov` and `solvax.gauss_newton_least_squares` expose an
  admissibility predicate but neither bound clipping nor the energy-merit
  globalization, so a migration would change the convergence contract, not
  only the algebra.
- **Cold and warm cost** — mirror solve compiles already fall below the
  benchmark harness's minimum-compile floor (M1 in
  `benchmarks/profile_workflows.py`), and the host Krylov loop is not the
  measured bottleneck; moving GMRES into JAX would add compilations to the
  lane least able to amortize them.
- **Dense rescue** — already a bounded small-case diagnostic, not an
  unbounded fallback.

The generic pieces SOLVAX does provide are used where they are generic:
`solvax.gmres` serves as the independent reference Krylov solver in the
preconditioner tests, and the forcing-term and pseudo-transient literature
the lane's tolerances follow is the same SOLVAX implements (Eisenstat and
Walker 1996; Kelley and Keyes 1998). Migration is deferred until a measured
comparison at production size shows equal-or-better cost under the same
contract.

## Validation suite map

The isotropic acceptance items and where each is checked:

- straight circular vacuum mirror — `test_isotropic_forces.py::test_vacuum_cylinder_has_exact_energy_and_negligible_physical_force`
- paraxial finite-beta mirror — `test_boundary_conditions.py::test_paraxial_finite_beta_solve_matches_long_thin_pressure_balance`
- manufactured geometry/field with known curl — `test_isotropic_forces.py::test_flared_tube_manufactured_lorentz_force_converges_spectrally`
- weak residual versus AD energy gradient — `test_isotropic_forces.py::test_staggered_first_variation_matches_autodiff_for_3d_finite_beta`
- strong residual convergence — `test_isotropic_forces.py::test_manufactured_radial_pressure_balance_converges_second_order` and the refinement ladders in `test_splines.py`
- axis regularity — `test_boundary_conditions.py` audit tests and `test_isotropic_forces.py::test_radius_interpolation_respects_odd_mode_axis_regularity`
- cut boundary terms — `test_boundary_conditions.py::test_cut_flux_directional_derivative_converges_to_natural_term`
- lateral fixed-boundary constraint — `test_boundary_conditions.py::test_lateral_and_cut_families_are_null_directions_of_the_projected_gradient`
- free-boundary interface residual — `test_boundary_conditions.py::test_interface_residual_measures_total_pressure_jump` and the beta-scan tests in `test_free_boundary.py`
- divergence-free field — `test_mirror_geometry_fields.py`
- Fourier versus B-spline axial representation — `test_qi_hybrid.py::test_bspline_reproduces_legs_better_than_fourier` and `test_splines.py::test_coefficient_native_state_matches_chebyshev_polynomial_geometry_and_energy`
- periodic hybrid limit — `test_splines.py::test_closed_circular_limit_reaches_ftol_with_independent_strong_force`

## References

- S. P. Hirshman and J. C. Whitson, *Steepest-descent moment method for
  three-dimensional magnetohydrodynamic equilibria*, Phys. Fluids 26, 3553
  (1983). <https://doi.org/10.1063/1.864116>
- P. Merkel, *An integral equation technique for the exterior and interior
  Neumann problem in toroidal regions*, J. Comput. Phys. 66, 83 (1986).
  <https://doi.org/10.1016/0021-9991(86)90055-0>
- S. P. Hirshman, W. I. van Rij and P. Merkel, *Three-dimensional free boundary
  calculations using a spectral Green's function method*, Comput. Phys. Commun.
  43, 143 (1986). <https://doi.org/10.1016/0010-4655(86)90058-5>
- M. G. Duffy, *Quadrature over a pyramid or cube of integrands with a
  singularity at a vertex*, SIAM J. Numer. Anal. 19, 1260 (1982).
  <https://doi.org/10.1137/0719090>
- W. A. Cooper et al., *3D magnetohydrodynamic equilibria with anisotropic
  pressure*, Comput. Phys. Commun. 72, 1 (1992).
  <https://doi.org/10.1016/0010-4655(92)90002-G>
- W. A. Cooper et al., *Three-dimensional anisotropic pressure free boundary
  equilibria*, Comput. Phys. Commun. 180, 1524 (2009).
  <https://doi.org/10.1016/j.cpc.2009.04.006>
- D. Endrizzi et al., *Physics basis for the Wisconsin HTS Axisymmetric
  Mirror (WHAM)*, J. Plasma Phys. 89 (2023).
  <https://doi.org/10.1017/S0022377823000806>
- S. J. Frank et al., *Nonlinear anisotropic equilibrium reconstruction in
  axisymmetric magnetic mirrors*. <https://arxiv.org/abs/2509.17288>
- High-order polar regularity for smooth polar charts.
  <https://arxiv.org/abs/2601.17841>
- S. C. Eisenstat and H. F. Walker, *Choosing the forcing terms in an
  inexact Newton method*, SIAM J. Sci. Comput. 17 (1996).
  <https://doi.org/10.1137/0917003>
- C. T. Kelley and D. E. Keyes, *Convergence analysis of pseudo-transient
  continuation*, SIAM J. Numer. Anal. 35, 508 (1998).
  <https://doi.org/10.1137/S0036142996304796>
