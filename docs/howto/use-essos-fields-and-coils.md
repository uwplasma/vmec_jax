# Use ESSOS fields and coils

VMEX and [ESSOS](https://github.com/uwplasma/ESSOS) meet at two Python seams,
and each one runs in a single direction. Neither package imports the other:
the equilibrium crosses as a wout, the coil field crosses as a tabulated
grid. Requires ESSOS (`pip install essos`).

| Direction | Call | Crosses as |
| --- | --- | --- |
| VMEX equilibrium to ESSOS | {func}`~vmex.core.tracing.essos_vmec_field` | wout tables |
| ESSOS coils to VMEX | {meth}`~vmex.core.mgrid.MgridField.from_coils` | cylindrical field grid |

There is no third seam. VMEX owns fixed and free-boundary equilibrium
physics and its diagnostics; ESSOS owns coil geometry, Biot-Savart fields,
particle and field-line tracing and loss diagnostics.

## Hand an equilibrium to ESSOS

```python
import vmex as vj

field = vj.essos_vmec_field("wout_case.nc")   # essos.fields.Vmec
print(field.nfp, field.AbsB([0.5, 0.0, 0.0]), field.surface.gamma.shape)
```

`essos_vmec_field` takes either an in-memory
{class}`~vmex.core.wout.WoutData` or a path to a `wout_*.nc`. Released ESSOS
reads a wout *file*, so an in-memory equilibrium is written to a temporary
wout; ESSOS loads every table eagerly in its constructor, so the file is
gone by the time the field is returned. Keyword arguments pass through to
`essos.fields.Vmec` (`ntheta`, `nphi`, `close`, `range_torus`), which set the
resolution of the `field.surface` ESSOS builds alongside the field.

Three consequences worth knowing before you build on this:

- **The write severs the gradient.** This seam is a diagnostic route, not a
  differentiable one. A differentiable alpha-loss objective needs the ESSOS
  array constructor (uwplasma/ESSOS#61) and is not available here.
- **Stellarator symmetry only.** Released ESSOS reads the symmetric wout
  tables, so an `lasym` equilibrium is rejected rather than silently
  half-transferred.
- **Radial resolution is yours to choose.** ESSOS interpolates the half-mesh
  tables linearly in `s`. On the bundled precise-QA case the two independent
  `|B|` channels — `AbsB` from `bmnc`, and `norm(B)` built from `bsub*`,
  `gmnc` and the geometry — differ by 5.9% at `ns = 16` and 1.7% at
  `ns = 51`. Solve on the grid your diagnostic needs.

The tables themselves cross exactly: rebuilding the last closed surface from
ESSOS' `to_xyz` reproduces VMEX's own Fourier sum to 2.2e-16 m.

For fusion alphas specifically, {func}`~vmex.core.tracing.trace_alphas` and
`vmex --trace` wrap this seam with the ensemble, the loss-fraction
diagnostics and the figures ({doc}`trace-alpha-particles`).

## Bring an ESSOS coil field back

VMEX keeps no coil code, and its free-boundary solver consumes only a
magnetic field, so an ESSOS coil set enters through one tabulation:

```python
from essos.coils import Coils

coils = Coils.from_json("coils.json")
coil_field = vj.MgridField.from_coils(coils)          # or pass a BiotSavart
res = vj.solve_free_boundary(inp, external_field=coil_field)
```

Unset bounds default to the coil bounding box grown by 10%, and `nfp`
defaults to the coil set's own period count. Pass `rmin`/`rmax`/`zmin`/`zmax`
to bracket the plasma more tightly than the coils do, and `ir`/`jz`/`kp` to
set the grid (96, 96, 32 by default). The result is the same in-memory
{class}`~vmex.core.mgrid.MgridField` the mgrid-file lane produces — no
temporary file — and it is reused across every radial stage and hot restart.
`NZETA` must divide `kp`, which is VMEC2000's `mgrid_mod` pairing rule.

Judge the grid where NESTOR uses it, on the plasma boundary. On the bundled
Landreman-Paul QA coil set at the default resolution, the interpolated field
matches direct Biot-Savart on the converged boundary to 2.7e-5 median and
1.1e-4 maximum relative error. Trilinear interpolation degrades within a few
centimetres of a filament, so a bounding box taken from the coils is a
sampling region, not an accuracy claim.

The same route backs the CLI, where the coils come from a file:

```console
vmex input.case --coils coils.json
```

Tabulation is host-side and keeps no derivative with respect to coil shape.
For coil-shape gradients use
{meth}`~vmex.core.mgrid.MgridField.from_parameterized_cartesian_field`, which
stays inside JAX, or the virtual-casing residual
({doc}`free-boundary`). The reverse of this seam — recovering coils from a
field grid — is a coil-design inverse problem and belongs in ESSOS, not
here.

## Walk both seams

`examples/vmex_essos_workflow.py` runs the round trip end to end: solve a
fixed boundary, hand it to ESSOS, measure the rotational transform from an
ESSOS field-line trace (`0.419167` traced against `0.419155` in the wout, a
relative 2.9e-5), tabulate the ESSOS coil set, solve a vacuum free boundary
with it, and hand that equilibrium back across the same seam. It takes two to
four minutes, or one to two under `VMEX_EXAMPLES_CI=1`.

`examples/free_boundary_essos_coils.py` is the dedicated pressure scan on
the coil-held plasma, and `examples/vmex_get_B_outside_plasma.py` queries the
exterior field with coils and virtual casing.
