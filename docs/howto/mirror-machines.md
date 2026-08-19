# Solve mirror and hybrid equilibria

`vmex.mirror` solves open-mirror equilibria (fixed and free boundary) and
closed stellarator-mirror hybrids in a spline-native basis. This page is the
run recipes; the theory, gate evidence, and lane status are in
{doc}`/explanation/mirror-geometry`, and the `mout_*.nc` output format is
{doc}`/reference/mout-file`.

## Solve a fixed-boundary mirror from one radius

The one-call entry point solves an axisymmetric open mirror from a single
LCFS radius, picking a default boundary and profiles for the requested
resolution:

```python
from vmex.mirror import MirrorConfig, MirrorResolution, solve_fixed_boundary_from_radius

config = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=4, nxi=17))
result = solve_fixed_boundary_from_radius(0.3, config)
```

`MirrorResolution` takes `ns` (radial surfaces), `mpol` (largest represented
poloidal Fourier mode; the collocation size is the read-only
`ntheta = 2*mpol + 1`), and `nxi` (axial nodes); the returned
`SplineMirrorSolveResult` carries the converged coefficients and the
variational, weak, and pointwise-force residuals.

For a shaped boundary, build `SplineMirrorBoundary` / `SplineMirrorState` /
`SplineMirrorDiscretization` directly and call
{func}`vmex.mirror.solve_fixed_boundary`.

## Run the shipped examples

Four runnable examples ship with the package and need no command-line
arguments (each has editable inputs at its top):

```console
python examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py   # rotating-ellipse fixed boundary
python examples/mirror/mirror_free_boundary_beta_scan.py          # axisymmetric free-boundary beta scan
python examples/mirror/stellarator_mirror_hybrid.py               # periodic B-spline racetrack hybrid
python examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py     # QI-mirror hybrid: Fourier vs B-spline
```

The first checks every convergence gate for the rotating ellipse and the
axisymmetric mirror, differentiates rotating-ellipse volume against two fully
reconverged solves, and writes MOUT plus 3-D, cross-section, `|B|`, residual,
symmetry, and analytic-direction figures.

## Plot the results

Open-mirror solves write mirror-native `mout_*.nc` files, which plot with

```console
vmex --plot mout_example.nc
```

rendering horizontal 3D, coil, cap-to-cap field-line, `|B|`, pressure,
cross-section, and residual figures.

## Run a free-boundary beta scan

```console
python examples/mirror/mirror_free_boundary_beta_scan.py
```

The script solves every beta point from 0 through 50% and writes one MOUT per
state, a compact JSON summary, restart files, and reviewed figures under
`results/mirror_free_boundary_beta_scan/`. The example's two ESSOS loops are
sized to the plasma: radius 0.5 m at z = +/-1.0 m carrying 3.72e5 A each,
reproducing the central vacuum field B(0) = 0.0836 T of the recorded
benchmark geometry with vacuum mirror ratio 4.58. The axisymmetric
free-boundary lane is supported through 10% requested beta; 25% and 50% are
extended validation ({doc}`/reference/capabilities`).

External fields enter as an ESSOS/SIMSOPT Biot-Savart object, any
vectorized `xyz -> B` callable, or a shared
{class}`~vmex.core.mgrid.MgridField`; coil geometry stays in ESSOS. Field
callables that capture committed arrays should use `jax.tree_util.Partial`
(or another registered pytree) so VMEX can relocate the captured leaves; an
ordinary Python closure is opaque and pins its arrays' placement.

### Resume an interrupted scan

Set `SAVE_RESTARTS = True` in the example to write one compressed `.npz`
hot-start per beta point
({func}`vmex.mirror.output.save_free_boundary_restart`). To resume, set
`RESTART_FROM` and trim `BETAS` to the unfinished suffix;
{func}`vmex.mirror.output.load_free_boundary_restart` checks the schema and
coefficient shapes before returning the boundary, plasma state, and
calibrated mass scale. The original beta-zero boundary remains the
pressure-profile reference.

## Build a hybrid

```python
from vmex.mirror import build_stellarator_mirror_hybrid, solve_fixed_boundary

setup = build_stellarator_mirror_hybrid(axis_coefficient_count=16)
result = solve_fixed_boundary(setup.discretization, setup.state)
```

`build_stellarator_mirror_hybrid` constructs the periodic B-spline racetrack
(two exactly straight mirror legs, two stellarator returns); pass
`axis_coefficient_count` to freeze the leg-return junction while refining the
solve basis — the contract under which the circular-section lane converges
monotonically ({doc}`/explanation/mirror-geometry`). `build_qi_mirror_hybrid`
splices straight legs into a QI stellarator axis instead;
`examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py` runs that construction end
to end.

## Pick the device

Mirror fixed/free-boundary solves and beta scans expose the same `device=`
contract as the toroidal core. On the office host, the corrected 15x15 case
took 35.2 s on CPU and 44.2 s on one RTX A4000, so the mirror-specific
`device="auto"` policy selects CPU for its SciPy-controlled JAX callbacks.
Explicit `device="cpu"`/`"gpu"` always wins and `device=None` follows
ordinary JAX placement; no environment variable is required.

## Differentiate a mirror equilibrium

`spline_fixed_boundary_adjoint` (scalar diagnostics, reverse) and
`spline_fixed_boundary_tangent` (forward) differentiate through the converged
coefficient residual; `free_boundary_adjoint` covers the axisymmetric
free-boundary lane through the 10% beta ceiling (validated to 1.1e-10
relative against reconverged finite differences). Scope and validation
evidence: {doc}`/explanation/mirror-geometry`.
