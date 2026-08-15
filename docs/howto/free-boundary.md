# Run a free-boundary equilibrium

Set `LFREEB = T` and supply the external field — a MAKEGRID `mgrid` file
with `EXTCUR` currents, or ESSOS coils directly — and VMEC finds the plasma
boundary that balances against it: the last closed flux surface is an
output, not an input.

## From an mgrid file

```text
&INDATA
  LFREEB = T
  MGRID_FILE = 'mgrid_case.nc'
  EXTCUR = 1.0e5  1.0e5  ...     ! one current per coil group
  NVACSKIP = 6                   ! full NESTOR solve cadence
  ...
```

```console
vmex input.case
```

The console shows the VMEC2000 free-boundary output (`In VACUUM` block,
`VACUUM PRESSURE TURNED ON` banner), and the wout gains the free-boundary
metadata (`nextcur`/`extcur`/`curlabel`/`mgrid_mode`) plus the NESTOR
potential and surface fields (`potsin`/`xmpot`/`xnpot`/`*_sur`). A missing
mgrid file falls back to a fixed-boundary solve with a warning — retained
VMEC2000 behavior, so check the banner if a "free-boundary" run looks
suspiciously fixed. End to end:

```{literalinclude} ../../examples/free_boundary_mgrid.py
:language: python
```

## From ESSOS coils (no mgrid file)

VMEX is coil-agnostic: the solver consumes only a magnetic field. Set
`MGRID_FILE = 'DIRECT_COILS'` and pass an ESSOS coils file
(requires `pip install essos`):

```console
vmex input.case --coils coils.json
```

The coils' Biot-Savart field is tabulated once into an in-memory
{class}`~vmex.core.mgrid.MgridField` via
{meth}`~vmex.core.mgrid.MgridField.from_cartesian_field` — no temporary
mgrid file. The same adapter accepts ESSOS' `B(points)` protocol, SIMSOPT's
`set_points(points); B()` protocol, or any plain Cartesian callable, and the
tabulated field is reused across every radial stage and hot restart.
`examples/free_boundary_essos_coils.py` runs a beta scan against the
Landreman-Paul precise-QA coil set this way:

```{literalinclude} ../../examples/free_boundary_essos_coils.py
:language: python
```

## Key knobs

- `EXTCUR` — coil-group currents scaling the mgrid field.
- `NVACSKIP` — iterations between full NESTOR solves; between them, cheap
  incremental updates reuse the factored potential matrix, and the cadence
  adapts toward convergence ({doc}`/explanation/nestor-vacuum`).
- `--jacobian-retries` — free-boundary recovery after the 75-reset condition
  rebuilds the axis filament and NESTOR structures before continuing
  ({doc}`troubleshoot`).

## Convergence differences from fixed boundary

The vacuum solve activates only once `fsqr + fsqz <= 1e-3`, so early
iterations run effectively fixed-boundary; expect the residual trace to
change character at the `VACUUM PRESSURE TURNED ON` banner. Free-boundary
ladders carry the active-vacuum state and adaptive `NVACSKIP` across
`NS_ARRAY` stages ({doc}`/explanation/multigrid`). On GPUs, the dense NESTOR
factor runs on CPU by design ({doc}`run-on-gpu`).

## Differentiability scope

The NESTOR fixed point is not differentiated. Coil/`extcur` gradients on a
specified boundary use the virtual-casing residual
(`examples/take_free_boundary_gradients.py`), while the compact fixed-boundary
plasma-and-coil examples are under `examples/optimization/`. Scope is stated
in {doc}`/reference/capabilities` and the mechanism in
{doc}`/explanation/nestor-vacuum`.
