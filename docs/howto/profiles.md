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

Every profile key, default, and accepted `*_TYPE` string:
{doc}`/reference/input-file` (Pressure profile / Current and iota sections).
The evaluation code is {mod}`vmex.core.profiles`.
