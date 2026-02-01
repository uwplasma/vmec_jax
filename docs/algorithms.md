# Algorithms

This page documents the numerical building blocks currently implemented, and the intended path to a
VMEC-quality solver.

## Discretization summary

### Radial grid

We use a uniform grid in `s ∈ [0,1]` with `ns` points:

\\[
s_j = \frac{j}{ns-1},\quad j=0,\dots,ns-1.
\\]

VMEC2000 uses a mix of full-mesh and half-mesh conventions in `s`; `vmec-jax` currently treats
most quantities on a full mesh for simplicity, and ports half-mesh logic as needed for parity.

### Angular grids

We use uniform tensor-product grids in `θ` and `ζ` (one field period):

- `θ_i = 2π i / ntheta`
- `ζ_k = 2π k / nzeta`

These grids are represented by `AngleGrid` and built in `vmec_jax/grids.py`.

### Fourier transforms

VMEC uses Fourier transforms between mode space and real space on the angular grid.
In `vmec-jax` we currently implement synthesis using dense basis tensors:

\\[
f(\theta,\zeta) = \sum_{m,n} \bigl(c_{mn}\cos(m\theta-n\zeta) + s_{mn}\sin(m\theta-n\zeta)\bigr).
\\]

This is implemented with `einsum` for both NumPy and JAX.

Future work:

- FFT-based transforms to reduce memory and runtime for larger mode counts.

## Step-1/2 geometry pipeline

The geometry kernel is the foundation of most downstream physics:

1. Evaluate `(R,Z,λ)` on `(s,θ,ζ)` grid from Fourier coefficients (`eval_coords`).
2. Compute angular derivatives analytically in mode space (`eval_fourier_dtheta`, `eval_fourier_dzeta_phys`).
3. Compute radial derivatives by finite differences on coefficient arrays (`d_ds_coeffs`), then re-synthesize.
4. Embed in Cartesian coordinates using `φ_phys = ζ/NFP`.
5. Compute covariant metric elements and signed Jacobian `sqrtg` (`eval_geom`).

The goal is that each step is differentiable and jittable.

## Step-3 profiles and volume integrals

### Profiles

VMEC supports several profile parameterizations. `vmec-jax` currently implements:

- `power_series` for pressure (`AM`), iota (`AI`), and current function (`AC`).

This is enough to validate many basic equilibria and match a subset of VMEC inputs.

Future work:

- pedestal logic parity (beyond the minimal clamp already implemented),
- spline and tabulated profiles,
- correct handling of `gamma != 0` “mass profile” pathway (pressure derived from volume profile).

### Volume profile

Given `sqrtg(s,θ,ζ)` we compute:

- `dV/ds` by integrating over angles,
- `V(s)` by a cumulative trapezoid in `s`.

This is implemented in `vmec_jax/integrals.py`.

## Step-4 field and energy

We compute contravariant field components `(bsupu, bsupv)` using:

- `sqrtg`,
- 1D flux functions `(phipf, chipf)`,
- scaled lambda derivatives multiplied by `lamscale`.

We validate against `wout` Nyquist Fourier coefficients for `sqrtg` and `bsup*`.

We then compute `B^2` using the covariant metric and integrate to obtain `wb`.

## Step-5 lambda solve (inner solve)

Holding `(R,Z)` fixed, we minimize `wb` with respect to lambda coefficients.
This is a useful subproblem and is part of VMEC’s nonlinear solve.

`vmec-jax` implements a robust baseline method:

- gradient descent in coefficient space,
- backtracking line search enforcing monotone decrease,
- gauge fixing of the `(m,n)=(0,0)` lambda mode.

This is implemented in `solve_lambda_gd`.

## Step-6/7 fixed-boundary solve (early stage)

We extend the optimization variables to include all Fourier coefficients:

- interior surfaces: evolve R/Z/λ,
- boundary surface (`s=1`): hold R/Z fixed (prescribed boundary),
- axis surface (`s=0`): enforce basic regularity by zeroing all `m>0` R/Z coefficients,
- lambda: enforce gauge and axis row constraints.

Two optimizers are currently provided:

1. Gradient descent + backtracking (`solve_fixed_boundary_gd`)
2. L-BFGS + backtracking (`solve_fixed_boundary_lbfgs`)

These are **not** yet VMEC-quality. In VMEC2000, the “right” approach is to:

- evaluate force residuals (not just energy),
- apply strong preconditioning in Fourier × radial space,
- use VMEC’s tailored time-stepping / Richardson-like evolution with Jacobian-based safeguards.

## Roadmap to VMEC-quality parity

The main missing pieces for parity are:

1. **Force residuals**:
   - full MHD force balance in VMEC coordinates,
   - correct half-mesh/full-mesh placement of terms.

2. **Preconditioning**:
   - mode-space block/diagonal scaling,
   - radial block-tridiagonal structure (VMEC’s strongest preconditioner).

3. **Nonlinear solve strategy**:
   - implement VMEC’s evolution strategy or a robust quasi-Newton with good preconditioner,
   - add Jacobian sign and axis-guess recovery logic.

4. **Implicit differentiation**:
   - replace backprop through iterations with a custom VJP based on the implicit function theorem.

