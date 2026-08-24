# Set pressure, current, and iota profiles

Profiles enter the deck as polynomial coefficients (`power_series`) or
spline knots (`cubic_spline` via the `*_AUX_S/*_AUX_F` arrays), and `NCURR`
selects whether you prescribe the rotational transform (`NCURR=0`) or the
toroidal current (`NCURR=1`).

## Pressure

```text
&INDATA
  PMASS_TYPE = 'power_series'
  AM = 1.0  -1.0                 ! p(s) = AM(0) + AM(1)*s + ...
  PRES_SCALE = 1.0e5             ! multiplies the whole profile
  GAMMA = 0.0                    ! 0 = prescribed pressure (the usual mode)
```

or as a spline:

```text
  PMASS_TYPE = 'cubic_spline'
  AM_AUX_S = 0.0  0.25  0.5  0.75  1.0    ! knot locations in s
  AM_AUX_F = 1.0  0.9   0.6  0.2   0.0    ! values at the knots
```

`PRES_SCALE` multiplies either representation, so profile *shape* and
*amplitude* stay separate — beta scans ramp `PRES_SCALE` and keep `AM`
fixed (`examples/finite_beta_scan.py`).

## Current or iota (`NCURR`)

```text
  NCURR = 1                      ! prescribed current
  PCURR_TYPE = 'power_series'
  AC = 1.0  -1.0                 ! I'(s) shape
  CURTOR = 1.0e6                 ! total toroidal current [A]
```

```text
  NCURR = 0                      ! prescribed iota
  PIOTA_TYPE = 'power_series'
  AI = 0.9  -0.65                ! iota(s) = AI(0) + AI(1)*s + ...
```

Both accept the `cubic_spline` form with `AC_AUX_S/AC_AUX_F` and
`AI_AUX_S/AI_AUX_F`. At `NCURR=1` the transform is an output — read `iotaf`
from the wout.

## Same equilibrium, both representations

`examples/profiles_power_and_spline.py` solves one deck with polynomial and
spline profiles and shows they agree:

```{literalinclude} ../../examples/profiles_power_and_spline.py
:language: python
```

## Pitfalls

- **`SPRES_PED`** — pressure pedestal location: pressure is held constant
  for `s > SPRES_PED`. The default 1.0 disables it; a stale value from a
  copied deck silently flattens your edge pressure.
- **`BLOAT`** — expands the current/iota profile domain; nonzero values
  change where profiles are evaluated. Leave at 1.0 unless you are matching
  a VMEC2000 run that used it.
- **Ascending coefficients.** `AM/AC/AI` are ascending powers of `s`
  (`AM(0)` the axis value), matching VMEC2000 and simsopt
  `ProfilePolynomial`.
- **Spline knots trim at the first non-increasing entry.** Following
  VMEC2000 `profile_functions.f`, the `*_AUX_S` vector is cut to its
  strictly increasing leading segment (unset entries default to -1), so a
  mis-ordered knot silently shortens the profile — check the parsed
  {class}`~vmex.core.input.VmecInput` if a spline looks truncated.

`sum_atan` is available for `PCURR_TYPE` and `PIOTA_TYPE` (VMEC2000 offers it
for those two and not for pressure). It prescribes the profile itself, not its
derivative:

$$
f(x) = c_0 + \frac{2}{\pi}\sum_{k=0}^{4} c_{1+4k}
  \arctan\left(\frac{c_{2+4k}\, x^{c_{3+4k}}}{(1-x)^{c_{4+4k}}}\right)
$$

using `AC[0:21]` or `AI[0:21]`. Each group of four is an amplitude, a scale,
and the two exponents, so one group already gives a tunable edge-localized
step — the shape used for tokamak-like current profiles that rise sharply
near the boundary. At `x >= 1` VMEC2000 substitutes the hardcoded sum
$c_0 + c_1 + c_5 + c_9 + c_{13} + c_{17}$, which is the limit only when the
scales and the $(1-x)$ exponents are positive; vmex reproduces that as
written.

vmex implements every parameterization VMEC2000 offers: ten for
`PMASS_TYPE`, seven for `PIOTA_TYPE`, seventeen for `PCURR_TYPE`. A name
outside those raises rather than falling back to a default.

One deliberate departure from the Fortran: in `two_power_gs`, a Gaussian peak
slot with zero amplitude is skipped. VMEC evaluates `exp(-((x-0)/0)**2)` for an
unset slot, which is harmless away from the axis but `0/0` at `s = 0`, so a
deck using one or two of the six peaks gets NaN there. Every peak that is
actually set is bit-identical to VMEC.

Every profile key, default, and accepted `*_TYPE` string:
{doc}`/reference/input-file` (Pressure profile / Current and iota sections).
The evaluation code is {mod}`vmex.core.profiles`.
