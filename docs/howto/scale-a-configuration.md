# Scale a configuration to reactor size

`vmex --scale` applies the ideal-MHD dimensional similarity transform to an
input deck or wout file — by explicit `B_scale R_scale` factors, or to the
ARIES-CS reference magnitudes `|b0| = 5.7 T`, `Aminor_p = 1.7 m` when no
factors are given. Use it for physical orbit studies where particle energy
and Larmor radius are fixed (e.g. 3.5 MeV alpha calculations).

## Scale a deck or a wout

```console
vmex --scale input.case            # target ARIES-CS |b0| and Aminor_p
vmex --scale input.case 1.2 0.8    # explicit B_scale=1.2, R_scale=0.8
vmex --scale wout_case.nc          # scale a finished equilibrium directly
```

Output names gain `_scaled`. A positive magnetic factor preserves the flux
direction. Then solve and plot the scaled deck as usual:

```console
vmex input.case_scaled --plot --booz
```

## What transforms how

Normalized flux, rotational transform, beta, aspect ratio, mode numbers, and
profile shapes do not change. Dimensional quantities scale as:

| quantity | factor |
|----------|--------|
| boundary, axis, major/minor radii | $R_s$ |
| volume, Jacobian Fourier coefficients | $R_s^3$ |
| magnetic field | $B_s$ |
| covariant field, total/coil currents | $B_s R_s$ |
| contravariant field | $B_s/R_s$ |
| toroidal and poloidal flux | $B_s R_s^2$ |
| pressure and $B^2$ | $B_s^2$ |
| magnetic/pressure energy | $B_s^2 R_s^3$ |
| $\mathbf{J}\cdot\mathbf{B}$ | $B_s^2/R_s$ |
| wout `DMerc`, `DShear`, `DWell`, `DCurr`, `DGeod` | $B_s^{-2}R_s^{-4}$ |
| `IonLarmor` | $B_s^{-1}$ |

For inputs, pressure scales once through `PRES_SCALE`; `AM`, `AM_AUX_F`,
current/iota profiles, and every normalized spline coordinate remain shape
data. `CURTOR` scales only in prescribed-current mode.

## ARIES-CS targets from an input

A wout contains `b0` and `Aminor_p`, so its factors are exact. A fixed
boundary gives `Aminor_p` directly by Fourier quadrature, but `b0` depends
on the converged internal field, so VMEX runs a bounded radial probe at
`ns <= 9` and `ns <= 17` (final probe `ftol = 1e-10`) rather than the full
ladder. The command prints both resolutions and the coarse-to-fine changes —
those changes are the declared target uncertainty.

## Free-boundary decks

The probe also determines the final minor radius, and the scaled input and
mgrid sidecar are written together. Per-ampere (`mgrid_mode = S`) field
tables scale as $R_s^{-1}$; raw tables scale as $B_s$, while their recorded
currents and `EXTCUR` scale as $B_s R_s$. Direct-coil inputs are rejected:
their geometry and currents must be scaled before field tabulation.

## The validation contract

The defining test is commutation: (1) solve the original input and scale its
wout; (2) scale the input (and mgrid when present) and reconverge it;
(3) compare every physical wout scalar, profile, Fourier coefficient,
Mercier term, and NESTOR potential/surface field. VMEX runs this check for
finite-pressure prescribed-current fixed boundary and for symmetric and
LASYM free-boundary NESTOR cases. The structured functions
{func}`vmex.core.scaling.scale_input`,
{func}`vmex.core.scaling.scale_mgrid`, and
{func}`vmex.core.scaling.scale_wout` use parsed objects; they never edit
namelist text by prefix.
