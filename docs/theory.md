# Theory and conventions

This page summarizes the conventions used in `vmec-jax`. The goal is *compatibility* with VMEC2000
(`wout_*.nc`) and with standard VMEC literature.

## Coordinates and angles

VMEC uses curvilinear coordinates on nested flux surfaces. In this repo we use:

- `s ∈ [0,1]`: normalized toroidal flux label (VMEC’s “radial” coordinate).
- `θ ∈ [0, 2π)`: poloidal angle.
- `ζ ∈ [0, 2π)`: **field-period toroidal angle** (VMEC internal coordinate).
- physical toroidal angle `φ_phys = ζ / NFP`, where `NFP` is the number of field periods.

Fourier phases are written as:

\\[
\text{phase}(m,n; \theta,\zeta) = m\theta - n\zeta
\\]

where `n` is the *field-period* toroidal mode number (VMEC stores `xn = n*NFP` in `wout`).

Derivatives w.r.t physical toroidal angle satisfy:

\\[
\frac{\partial}{\partial \phi_{\mathrm{phys}}} = NFP \, \frac{\partial}{\partial \zeta}.
\\]

## Surface representation

VMEC represents a surface in cylindrical coordinates using Fourier series:

\\[
R(s,\theta,\zeta) = \sum_{m,n} \bigl( R_{mn}^c(s)\cos(m\theta-n\zeta) + R_{mn}^s(s)\sin(m\theta-n\zeta) \bigr),
\\]

\\[
Z(s,\theta,\zeta) = \sum_{m,n} \bigl( Z_{mn}^c(s)\cos(m\theta-n\zeta) + Z_{mn}^s(s)\sin(m\theta-n\zeta) \bigr).
\\]

`vmec-jax` stores these coefficients in a `VMECState` as arrays shaped `(ns, K)` where `K` is the
number of `(m,n)` modes in the main VMEC ordering.

## The lambda field

VMEC introduces a scalar field `λ(s,θ,ζ)` to define the “straight-field-line” poloidal angle:

\\[
u = \theta + \lambda(s,\theta,\zeta),
\\]

so that magnetic field lines are straight in the `(u,ζ)` angle pair.

### Important scaling convention

In VMEC2000 `wout` files, the stored Fourier coefficients of `λ` are **scaled** by a run-dependent
scalar `lamscale`. VMEC multiplies `∂λ/∂θ` and `∂λ/∂ζ` by `lamscale` before using them in the
contravariant field formulas.

`vmec-jax` follows this convention so we can validate against `wout` values.

## Geometry, metric, and Jacobian

We form covariant basis vectors by embedding the surface into 3D Cartesian coordinates using the
physical toroidal angle `φ_phys`:

\\[
x = R\cos\phi_{\mathrm{phys}}, \quad y = R\sin\phi_{\mathrm{phys}}, \quad z = Z.
\\]

The covariant basis vectors are:

\\[
\mathbf{e}_s = \partial_s \mathbf{r},\quad
\mathbf{e}_\theta = \partial_\theta \mathbf{r},\quad
\mathbf{e}_\phi = \partial_{\phi_{\mathrm{phys}}}\mathbf{r}.
\\]

The covariant metric is:

\\[
g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j,\quad i,j \in \{s,\theta,\phi\}.
\\]

The signed Jacobian is:

\\[
\sqrt{g} = \mathbf{e}_s \cdot (\mathbf{e}_\theta \times \mathbf{e}_\phi).
\\]

VMEC stores a sign convention `signgs = ±1` such that `signgs*sqrtg` is positive away from the axis.

## Contravariant magnetic field (VMEC form)

For fixed-boundary VMEC, the magnetic field is represented in terms of contravariant components
`B^u` and `B^v` (VMEC’s `bsupu` and `bsupv`) and 1D flux functions:

- `phipf(s) = dΦ/ds` (toroidal flux derivative),
- `chipf(s) = dΧ/ds` (poloidal flux derivative),

plus the lambda derivatives.

In this repo (matching VMEC’s `bcovar` + `add_fluxes` logic), the formulas are:

\\[
\mathrm{bsupv} = \frac{\mathrm{phipf} + \mathrm{lamscale}\,\partial_\theta\lambda}{\mathrm{signgs}\,\sqrt{g}\,2\pi},
\\]

\\[
\mathrm{bsupu} = \frac{\mathrm{chipf} - \mathrm{lamscale}\,\partial_\zeta\lambda}{\mathrm{signgs}\,\sqrt{g}\,2\pi}.
\\]

Note that `∂_ζ λ` is with respect to the field-period coordinate `ζ`, while the geometry kernel
returns `∂_{φ_phys} λ`, so we convert using `∂_ζ = (1/NFP) ∂_{φ_phys}`.

## Magnetic energy (`wb`)

VMEC reports the magnetic energy scalar `wb` in `wout` files. In VMEC normalization:

\\[
\mathrm{wb} = \frac{1}{(2\pi)^2}\int \frac{1}{2}|\mathbf{B}|^2 \, dV
\\]

where the integral is over the **full torus**.

In `vmec-jax`, we compute `B^2` from the contravariant components and the covariant metric:

\\[
B^2 = g_{\theta\theta}(\mathrm{bsupu})^2 + 2g_{\theta\phi}\mathrm{bsupu}\,\mathrm{bsupv} + g_{\phi\phi}(\mathrm{bsupv})^2.
\\]

## Thermal term (`wp`) and the total energy

VMEC also reports a thermal term `wp` (for finite pressure). A common VMEC “total energy” is:

\\[
W = \mathrm{wb} + \frac{\mathrm{wp}}{\gamma - 1},
\\]

with `γ` the adiabatic index. Many VMEC inputs use `gamma = 0`, which indicates a prescribed pressure
profile and is handled specially inside VMEC2000.

`vmec-jax` currently includes a simple `wp` term computed from an input pressure profile; full VMEC
pressure/mass handling (including the `gamma != 0` mass profile pathway) is part of future work.

