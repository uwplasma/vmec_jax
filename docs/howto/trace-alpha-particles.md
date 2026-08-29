# Trace alpha particles

`vmex --trace` follows an ensemble of fusion-born alpha particles
(guiding-centre model, ESSOS tracer) through a converged equilibrium and
reports the exact loss fraction; the same trace is one call away in Python
via {func}`~vmex.core.tracing.trace_alphas`. Requires ESSOS
(`pip install essos`).

## From the CLI

```console
vmex --trace wout_case.nc                  # trace, print, four figures
vmex input.case --trace                    # solve first, then trace
vmex --trace wout_case.nc --outdir figs/ \
     --trace-particles 400 --trace-tmax 1e-3 --trace-s 0.3
```

The console output gives the loss fraction, the lost / axis-termination /
solver-failure counts, and the tracing wall time. Four figures are written
next to the input (or into `--outdir`): `*_trace_trajectories.png` (sampled
orbits in 3-D over a translucent LCFS), `*_trace_vparallel.png`
(normalized parallel velocity), `*_trace_loss_fraction.png` (cumulative
loss fraction against time), and `*_trace_energy_error.png` (relative
energy error of the integrator).

Particles start on one flux surface `s` (uniform in poloidal angle, one
field period in toroidal angle, uniform pitch), at the fusion-alpha birth
energy of 3.52 MeV. An orbit counts as lost when it reaches `s >= 0.99`.
Loss fractions are physically meaningful at reactor scale — run
`vmex --scale wout_case.nc` first to put the equilibrium at ARIES-CS field
and size.

## From Python

```python
import vmex as vj

result = vj.trace_alphas("wout_case.nc", nparticles=400, tmax=1e-3)
print(result.loss_fraction, result.particles_lost)
vj.plot_tracing("wout_case.nc", result, outdir="figs")
```

`trace_alphas` accepts a path or an in-memory
{class}`~vmex.core.wout.WoutData` (written through a temporary wout file —
the route released ESSOS reads) and returns an
{class}`~vmex.core.tracing.AlphaTracingResult` with the loss-fraction time
series, per-particle loss times, trajectories in flux and Cartesian
coordinates, energies, and the counts.

For anything else ESSOS does with an equilibrium — field lines, surfaces,
`|B|` queries — {func}`~vmex.core.tracing.essos_vmec_field` returns the bare
`essos.fields.Vmec` this tracer is built on
({doc}`use-essos-fields-and-coils`).

## Scope

This is the exact loss-fraction *diagnostic*; it is parity with the rest of
the ecosystem — DESC v0.17 also ships particle tracing
(`desc.particles.trace_particles`). The differentiable alpha-loss
*objective* (a smooth surrogate a boundary optimization can descend) is a
separate feature that waits on the ESSOS array-based field constructor
(uwplasma/ESSOS#61) and vmex's traceable field tables. The exact loss
fraction is piecewise constant in the boundary — use it to certify, not to
optimize.
