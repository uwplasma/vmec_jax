# Set up a boundary optimization

{func}`vmex.core.optimize.least_squares` drives
`scipy.optimize.least_squares` over the boundary Fourier dofs with
simsopt-style `(function, target, weight)` terms and exact implicit
Jacobians — measured: precise QA (QS residual 7.2e-6) in one 14.5-minute
CPU call from a near-circular seed. This page is the campaign recipe;
{doc}`/tutorials/first-optimization` is the lesson, and the gradient
machinery is {doc}`/explanation/adjoint-gradients`.

## The driver

```python
import numpy as np
import vmex as vj
from vmex import optimize as opt

inp = vj.VmecInput.from_file("input.minimal_seed_nfp2")
qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10),
                                    helicity_m=1, helicity_n=0)   # QA
result = opt.least_squares(
    [(qs, 0.0, 1.0),
     (opt.aspect_ratio, 6.0, 1.0),
     (opt.mean_iota, 0.42, 1.0)],
    inp, max_mode=5,
    jac="implicit",       # exact implicit-differentiation Jacobians
    use_ess=True)         # spectral trust-region scaling (below)
result.input.to_indata("input.QA_optimized")
```

`RBC(0,0)` stays fixed; `max_mode` bounds the released harmonics. Repeated
trial solves are cheap by construction: executables are cached per
structure, every trial hot-restarts from the previous converged state
(sharpened by the perturbation warm start), and a failed trial returns a
large finite residual so the trust region backs off instead of crashing.

## Choose the objectives

Any mix of terms from {doc}`/reference/objectives`: quasisymmetry
(`QuasisymmetryRatioResidual`), omnigenity/QI (`QIResidual`,
`quasi_isodynamic_residual`), geometry (`aspect_ratio`, `volume`,
`mirror_ratio`), transform (`mean_iota`, `edge_iota`), stability
(`magnetic_well`, `mercier_stability_residual`, ballooning), bootstrap
(`RedlBootstrapMismatch`), turbulence proxies. The catalog marks which
objectives are traceable (usable with `jac="implicit"`) and which are
wout-engine only.

## Pick the gradient mode

- `jac="implicit"` — exact residual Jacobians by implicit differentiation:
  one amortized linear-algebra pass instead of ~2N solves. Requires
  traceable terms and a fixed-boundary problem. On the flagship campaigns
  implicit gradients are not merely faster — the exact-axisymmetric seed is
  a saddle of the QS residual where finite differences stall.
- `jac=None` — scipy `"2-point"` finite differences: one full solve per dof
  per Jacobian, works with *every* objective.
- `current_dofs=k` frees the first `k` current-profile (`AC`) coefficients
  plus `CURTOR` in either mode — the dof set of the self-consistent
  bootstrap objective.

## One call with ESS (the quickest survey pattern)

Instead of a `max_mode = 1, 2, ...` continuation ladder, hand the optimizer
**all** harmonics at once with Exponential Spectral Scaling (`use_ess=True`):
each dof's trust radius is scaled by
$e^{-\alpha \max(|m|,|n|)}/e^{-\alpha}$, so at the default `ess_alpha=1.2` a
`max_mode`-6 dof moves ~400x more cautiously than a `max_mode`-1 dof — the
coarse-to-fine ordering the ladder enforced, with no stage boundaries for
the objective to stall at. ESS improves trust-region scaling; it does not
reproduce the basin selection of a mode ladder.

Measured from a near-circular torus seed on a 36-core CPU
(`examples/optimization/QA_optimization_ess.py`, `QI_optimization_ess.py`):

| class | nfp | residual | seed | achieved | max_mode | dofs | wall |
|-------|-----|----------|------|----------|----------|------|------|
| QA | 2 | QS (1,0) | 2.04e-01 | **7.2e-06** | 5 | 120 | **14.5 min** |
| QI | 1 | omnigenity | 4.52e-01 | **1.81e-02** (25x) | 6 | 168 | **17.3 min** |

The staged ladder (`max_mode=(1, ..., 5)`) remains available — QA at QS
3.7e-7 in 25.5 min, ~1.8x longer — via `QA_optimization.py`,
`QH_optimization.py`, `QP_optimization.py`, `QI_optimization.py`
(constructed QI with a short quasi-poloidal basin stage first). A ladder
can reach a lower, different minimum even when the single-call pattern is
faster; the scripts stay side by side so the comparison is reproducible.

```{figure} /_static/figures/ess_x_scale.png
:alt: ESS trust-region scale versus harmonic level for alpha 0.7 and 1.2
:width: 78%

The ESS trust-region weight per harmonic level. Regenerate with
`python docs/_static/figures/sources/make_optimization_docs_figures.py`.
```

## Bounded-memory profile objectives (`minimize`)

Vector profile objectives normally need their complete residual Jacobian for
a Gauss-Newton step; at high resolution those radial block-tridiagonal
factors can dominate memory. {func}`vmex.core.optimize.minimize` minimizes
the identical scalar cost $\Phi(p) = \tfrac{1}{2}\sum_i r_i(p)^2$ with scipy
L-BFGS-B and one matrix-free reverse adjoint per gradient — independent of
the number of profile samples:

```python
result = opt.minimize(
    [(opt.mercier_stability_residual, 0.0, 1.0),
     (opt.glasser_stability_residual, 0.0, 1.0),
     (opt.jdotb_residual, 0.0, 1e-6)],
    inp, max_mode=5, bounds=bounds, options={"maxiter": 100})
```

Opt-in because it changes the step model from Gauss-Newton trust-region to
limited-memory quasi-Newton; the objective and its unconstrained minimizers
are unchanged. The perturbation warm start is unavailable on this path.

## Choose the optimizer

VMEX exposes the objective and derivatives without owning the optimizer
({doc}`/reference/optimization`). `examples/optimization/qi_shared_problem.py`
builds one QI problem that `QI_optimization_scipy.py`,
`QI_optimization_jaxopt.py`, and `QI_optimization_optax.py` send unchanged to
SciPy, JAXopt, or Optax. Install the optional JAX backends with
`pip install "vmex[optimizers]"`.

## Knobs that are already right by default

`jac_solver="auto"` (adjoint for scalar residuals, block-tridiagonal factor
for vector ones — 33x on the benchmark Jacobian phase),
`warm_start="perturbation"` (3.7x fewer forward iterations over 20 trials),
converged-state memo, `jac_chunk_size="auto"` memory bounding. What each one
does, with the measurements: {doc}`/explanation/adjoint-gradients`. Krylov
recycling (`recycle=True`) is off for a measured reason — solvax v0.1's FIFO
recycle space slows warm-started columns 1.7-3.4x; benchmark before
enabling.

## Reproduce the flagship campaigns

The committed decks in `benchmarks/opt_decks/`
(`input.{qa,qh,qi,qp}_{seed,optimized}`) pin the seeds and results: QA QS
7.2e-6 (single call) / 3.7e-7 (ladder), QH QS 5.83e-5, QP QS 3.3e-2 (an
extended ladder plus warm-start refinement; the hardest, basin-limited
class), QI omnigenity 1.81e-2. Figures:
`benchmarks/make_readme_figures.py --only optimization` and `--only qi`.

A different loop closes the current profile instead of the boundary:
`examples/optimization/QA_bootstrap_selfconsistent.py` erases the deck's
current profile and lets {func}`~vmex.core.bootstrap.self_consistent_bootstrap`
rebuild it from the Redl formula, converging to the
Landreman-Buller-Drevlak mismatch `f_boot = 2e-6` in a handful of
hot-restarted iterations ({doc}`/explanation/confinement`).
