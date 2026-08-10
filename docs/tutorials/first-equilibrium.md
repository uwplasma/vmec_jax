# Solve your first equilibrium

In this lesson you verify the install, solve a circular tokamak, and read
the solver's output. Twenty minutes, nothing but a terminal.

## Check the install

```console
vmex --test
```

This solves the bundled quasi-helically-symmetric stellarator case
(`input.nfp4_QH_warm_start`) end to end, writes `wout_nfp4_QH_warm_start.nc`
and diagnostic figures under `./vmex_test/`, and prints the equivalent
manual commands. If it finishes with figures on disk, everything works.

## Run an input file

`vmex` behaves like the `xvmec2000` executable: point it at a VMEC input
file. Use the circular tokamak deck from the repository
(`examples/data/input.circular_tokamak`, or any VMEC input you have):

```console
vmex input.circular_tokamak
```

The iteration table appears immediately:

```text
  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD
    1  3.94E-03  1.55E-03  1.22E-06  6.132E+00  9.00E-01  6.8059E+03
  200  1.65E-11  1.30E-11  1.27E-12  6.132E+00  9.00E-01  6.8059E+03
  368  8.72E-15  8.18E-15  3.53E-15  6.132E+00  9.00E-01  6.8059E+03
```

`FSQR/FSQZ/FSQL` are the squared force-balance residual norms in the radial,
vertical, and stream-function directions; the run stops when all three fall
below the deck's `FTOL_ARRAY` tolerance (here 1e-14). `RAX` is the magnetic
axis position, `DELT` the adaptive time step, `WMHD` the MHD energy the
iteration descends ({doc}`/explanation/variational-problem`).

Then the summary block:

```text
 Aspect Ratio          =       3.000000
 Plasma Volume         =     473.741011 [M**3]
 Major Radius          =       6.000000 [M]
 Minor Radius          =       2.000000 [M]
 |B| on Axis (b0)      =       5.241674 [T]
 ...
 NUMBER OF JACOBIAN RESETS =    0

 Wrote WOUT file: wout_circular_tokamak.nc
```

Zero Jacobian resets means the solver never had to back off from a
self-intersecting surface guess. The whole run takes a few seconds on a
laptop CPU.

## What you produced

`wout_circular_tokamak.nc` is a standard VMEC2000 output file: geometry as
Fourier tables, profiles, and scalars ({doc}`/reference/wout-file`). It loads
in simsopt, booz_xform, and any other VMEC-ecosystem tool — and in VMEX
itself:

```python
from vmex.core.wout import read_wout

wout = read_wout("wout_circular_tokamak.nc")
print("aspect ratio:", float(wout.aspect))
print("edge iota:   ", float(wout.iotaf[-1]))
```

## The same run as a script

`examples/fixed_boundary_run.py` is this lesson as a runnable script — read
a deck, solve on the multigrid ladder, write and plot the wout:

```{literalinclude} ../../examples/fixed_boundary_run.py
:language: python
```

Next: {doc}`plots-and-boozer` turns the wout into figures.
