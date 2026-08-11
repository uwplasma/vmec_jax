# Use results in simsopt, booz_xform, and your code

VMEX wout files implement the VMEC2000 netCDF schema — names, dimensions,
dtypes, unit conventions — so simsopt, booz_xform, and other VMEC-ecosystem
tools load them unchanged. This page shows the three access routes.

## Read a wout in Python

```python
from vmex.core.wout import read_wout

wout = read_wout("wout_case.nc")
print("aspect ratio:", float(wout.aspect))
print("edge iota:   ", float(wout.iotaf[-1]))
print("beta total:  ", float(wout.betatotal))
```

{class}`~vmex.core.wout.WoutData` exposes every schema variable as an
attribute: scalars (`aspect`, `b0`, `volume_p`, ...), radial profiles
(`iotaf`, `presf`, `jcurv`, `DMerc`, ...), and the Fourier tables (`rmnc`,
`zmns`, `lmns`, `bmnc`, ...). Field access follows the wout conventions —
`lmns` is half-mesh, `bsubsmns` full-mesh, `presf` in Pa. The complete
variable list with mesh/unit notes is {doc}`/reference/wout-file`.

Reconstruct fields on a grid from the Fourier tables, e.g. `|B|` on the
boundary from `bmnc` with phase `xm*theta - xn*zeta` (`xn` already includes
`nfp`); {mod}`vmex.core.plotting` contains worked reconstructions to crib
from (`boozer_modB_on_surface`, the surface synthesis in `plot_modB`).

## Load in simsopt

```python
from simsopt.mhd import Vmec

vmec = Vmec("wout_case.nc")            # wout mode: no Fortran VMEC needed
print(vmec.aspect(), vmec.mean_iota())
```

simsopt's `Vmec` wout-file mode, `SurfaceRZFourier.from_wout`, and the
Boozer/QS diagnostics all consume VMEX files directly — the parity tests pin
per-variable agreement against VMEC2000 golden runs
({doc}`/reference/performance`).

## Boozer transform

No external tool needed — `run_booz_xform` ships in the box:

```python
import vmex as vj

boozmn_path = vj.run_booz_xform("wout_case.nc", mbooz=48, nbooz=48)
vj.plot_boozmn(boozmn_path, outdir=".")
```

or from the CLI: `vmex wout_case.nc --booz` writes a standard `boozmn_*.nc`
that the C++ `booz_xform` tooling also reads
({doc}`/tutorials/plots-and-boozer`).

## Two things downstream readers should know

- **VMEX extension variables.** `vmex_diagnostics_schema = 1` and
  `vmex_trapped_fraction` (the effective trapped-particle fraction on the
  full mesh) are extra names VMEC2000 readers may ignore
  ({doc}`/reference/wout-file`).
- **Declared but fill-valued variables.** A schema variable may be
  fill-valued where its producer is not implemented; the disclosed cases are
  in {doc}`/reference/wout-file` and
  {doc}`/reference/vmec2000-compatibility`.
