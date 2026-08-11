# See the equilibrium

In this lesson you turn a `wout_*.nc` file into the standard figure set and
run the Boozer transform — both ship in the plain install.

## Plot a wout file

Every `wout_*.nc` (from VMEX or from VMEC2000 itself) plots directly:

```console
vmex --plot wout_circular_tokamak.nc
vmex input.circular_tokamak --plot     # solve, then plot in one command
```

Six PNG files appear next to the file (or in `--outdir`):

| file | contents |
|------|----------|
| `*_summary.png` | the diagnostic summary panel |
| `*_surfaces.png` | flux-surface cross-sections at several toroidal angles |
| `*_modB.png` | `\|B\|` contours in (zeta, theta) at mid radius and boundary |
| `*_profiles.png` | iota / pressure / current profiles + the `fsqt` convergence trace |
| `*_stability.png` | Mercier decomposition + frozen-equilibrium pressure scan |
| `*_boundary3d.png` | 3-D plasma boundary colored by `\|B\|` |

The summary panel is the one to look at first: rotational transform,
pressure, parallel current, stability profiles, a polar second-invariant map,
3-D LCFS, and Boozer `|B|` at a glance. The separate `*_surfaces.png` always
contains the toroidal cross-sections. Which panels it contains (and how to
plot from Python, select figures, or plot mirror `mout_*.nc` files) is
{doc}`/howto/plot-diagnostics`.

The pressure scan is a fast diagnostic of the explicit pressure-gradient
terms, not a substitute for a finite-pressure equilibrium sequence. It uses
the stored pressure shape, or a labeled linear seed for a vacuum WOUT.

## Boozer coordinates

The plain install includes the differentiable `booz_xform_jax` transform:

```console
vmex input.nfp4_QH_warm_start --booz          # solve + Boozer transform
vmex wout_nfp4_QH_warm_start.nc --booz        # transform an existing wout
vmex --plot boozmn_nfp4_QH_warm_start.nc      # Boozer |B| contours + spectra
```

`--plot` already performs the in-process transform needed for its Boozer
`|B|` panels. `--booz` is only needed when you want to write a standard
`boozmn_*.nc` file for later analysis. In Boozer coordinates field
strength contours reveal the symmetry class directly: for this
quasi-helically-symmetric case the `|B|` contours run diagonally — helical
symmetry — which is what makes the Boozer view the standard way to judge
quasisymmetry. The transform resolution and surfaces are configurable:

```console
vmex wout_nfp4_QH_warm_start.nc --booz --mbooz 48 --nbooz 48 \
     --booz-surfaces "0.25, 0.5, 1.0"
```

## The same run as a script

`examples/plot_and_boozer.py` produces every `plot_wout` figure and the
Boozer `|B|` spectrum on the last closed flux surface from Python:

```{literalinclude} ../../examples/plot_and_boozer.py
:language: python
```

Next: {doc}`first-gradient` differentiates the equilibrium you just plotted.
