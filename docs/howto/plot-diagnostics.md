# Plot equilibrium diagnostics

`vmex --plot` renders the standard figure set from any `wout_*.nc`,
`boozmn_*.nc`, or mirror `mout_*.nc` file; the same figures are one call
away in Python via {func}`~vmex.core.plotting.plot_wout` /
{func}`~vmex.core.plotting.plot_boozmn`.

## From the CLI

```console
vmex --plot wout_case.nc               # six wout figures
vmex input.case --plot                 # solve, then plot
vmex --plot boozmn_case.nc             # Boozer contours + spectra
vmex --plot mout_case.nc               # straight-axis mirror figures
vmex --plot wout_case.nc --outdir figs/
```

The wout set always includes `*_surfaces.png` (cross-sections at several
zeta over one field period, axis marked), `*_modB.png` (`|B|` contours at
mid radius and boundary), `*_profiles.png` (iota/pressure/current plus the
`fsqt` force-residual trace), `*_boundary3d.png` (3-D boundary colored by
`|B|`), `*_stability.png` (Mercier terms and a pressure scan), and
`*_summary.png`. Both symmetric and `lasym` equilibria are supported — the
sine/cosine partner tables are included whenever present. All figures use
the Agg backend at dpi >= 200, so plotting works on headless machines.

## From Python

```python
import vmex as vj

paths = vj.plot_wout("wout_case.nc", outdir="figs")        # dict[str, Path]
paths = vj.plot_wout(wout_data, outdir="figs",
                     which=("summary", "profiles"))        # select figures
```

`plot_wout` accepts a path or an in-memory
{class}`~vmex.core.wout.WoutData`, and `which=` selects a subset of
`("summary", "surfaces", "modB", "profiles", "stability", "3d")`.
Per-figure helpers
({func}`~vmex.core.plotting.plot_summary`,
{func}`~vmex.core.plotting.plot_stability`, ...) return single figures for
embedding in your own scripts; `examples/plot_and_boozer.py` is the worked
version.

## The summary figure

`*_summary.png` is a publication-style diagnostic set: rotational transform
(full mesh) with the parallel bootstrap current
$\langle \mathbf{J}\cdot\mathbf{B} \rangle$ on its right axis, a combined
confinement panel — pressure on the left axis, with the effective ripple
$\epsilon_{\rm eff}^{3/2}$ (NEO_JAX at the bounded
{func}`~vmex.core.neoclassical.diagnostic_neo_config` resolution) and the
fast-ion proxy $\Gamma_c$
({func}`~vmex.core.gammac.gamma_c_from_wout` at a compact radial-trend
sampling) sharing one dimensionless right axis — the relative radial
force error
$\epsilon_F=|(\mathbf J\times\mathbf B-\nabla p)_s|/(|(\mathbf J\times\mathbf B)_s|+|(\nabla p)_s|)$,
Mercier `DMerc` and the Glasser
resistive-interchange $D_R$ with $V''(s)$ on a color-matched right axis,
a 3-D LCFS, and the second adiabatic invariant in the polar disk
$x=s\cos\alpha$, $y=s\sin\alpha$. Concentric $J$ contours diagnose
alpha-independence. `|B|` in Boozer coordinates appears at mid radius and on
the LCFS as unfilled jet contours with a field line of slope iota. The Boozer
transform runs in-process, so `--plot` needs no separate `--booz` pass;
`--booz` is for writing a reusable `boozmn_*.nc`. The plotted $D_R$ comes from
a host-side reconstruction of the WOUT file, which carries the sine-parity
partner tables and so covers both symmetry classes. The reconstruction checks
itself by reproducing the stored `DMerc` profile from the same integrals; on
mismatch the curve is omitted with a panel note rather than drawn
unvalidated. The force-error maximum in the scalar card uses the solved
interior surfaces; WOUT's axis and boundary values are extrapolations. The
normalization breaks down in vacuum: with $\nabla p=0$ and no net current,
both the numerator and every term of the denominator
({func}`~vmex.core.postprocess.force_balance`) are radial differences of
near-constant `buco`/`bvco`, so the ratio is $O(1)$ discretization noise
rather than a force error. Read the panel only for finite-pressure or
current-carrying cases.
The confinement panel's two right-axis profiles are radial *trends*, not
transport numbers: both diagnostics run at bounded summary resolution, they
share the summary's one in-process Boozer transform where the mathematics is
common (NEO consumes it directly; $\Gamma_c$ keeps its validated real-space
field-line route), and the computed profiles are cached per in-memory WOUT so
repeated summary generation does not recompute or recompile them.  A
diagnostic that is unavailable (NEO_JAX not installed, `lasym` Boozer
tables, a failed evaluation) is dropped from the panel with the reason
recorded — never drawn as zero.  Use
{func}`vmex.core.neoclassical.epsilon_effective_from_wout` with an explicit
`neo_jax.NeoConfig` when a publication effective-ripple calculation is
wanted; the library default no longer clears process-wide JAX caches
(`clear_jax_caches=False`) — the CLI releases the plot-only executables
itself after all requested diagnostics.
The two stability indices and $V''(s)$ use separate scales whose zero levels
are aligned; $V''(s)<0$ denotes a magnetic well. Their legend sits below the
panel so it cannot hide a curve.

`*_stability.png` first separates `DMerc` into shear, well, current, and
geodesic terms. Its second panel rescales the WOUT pressure profile and plots
the worst frozen-equilibrium margins, $\min_s D_{Merc}$ and $-\max_s D_R$,
against trial volume-average beta; positive is favorable. A vacuum WOUT has no
pressure shape, so this panel states that it uses $p(s)\propto1-s$. This scan
isolates the explicit pressure-gradient drive at fixed geometry and current;
finite-pressure stability must still be certified by re-solving each point.

## Boozer figures

`vmex --plot boozmn_case.nc` renders `|B|` contours on the transformed
surfaces, the Boozer spectrum, and mode-amplitude profiles
({func}`~vmex.core.plotting.plot_boozmn_modB`,
{func}`~vmex.core.plotting.plot_boozmn_spectrum`,
{func}`~vmex.core.plotting.plot_boozmn_mode_profiles`). Producing the
`boozmn_*.nc` file in the first place is {doc}`/tutorials/plots-and-boozer`.

## Mirror figures

`vmex --plot mout_case.nc` renders the open-mirror set: horizontal 3D, coil
curves, cap-to-cap field lines, `|B|`, pressure, cross-sections, and
residual histories ({doc}`/reference/mout-file`).
