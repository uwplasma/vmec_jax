# Force-balance polishing measured without the saturating metric (2026-09-03)

Records: `polish_force_error_2026-09-03.json` (shaped tokamak) and
`polish_force_error_solovev_2026-09-03.json` (bundled solovev).
Both are `benchmarks/strong_polish.py` output, verbatim, each carrying the
deck hash, the measurement commit and clean flag, the versions, and the
invocation that produced it.

Protocol: one run at a time, foreground, from a clean checkout of the
measurement commit; the exact command is in each record's `command` field.
`initial_certificate` and `final_certificate` are the *same* lifted native
state before and after the correction — one spline basis, one set of
certificate nodes — so the pair is like for like and carries no export-mesh
or reconstruction difference.

## Why the records exist (plan 31.2-R1, 31.2-R3)

The shipped acceptance metric

    eps_F = 2|F| / (|JxB| + |grad p| + F_floor),   F = JxB - grad p

is bounded above by 2 by construction, because `|F| <= |JxB| + |grad p|`
pointwise. It saturates wherever the denominator collapses, and it is an L2
of a *pointwise ratio*, so it is dominated by whichever region has the
smallest denominator — usually the near-axis region. Neither property is
visible in the number itself. Each record therefore also carries the
volume-averaged relative force error of Panici et al. 2023, the vacuum-safe
`<|F|>/<|grad(B^2/2mu0)|>`, the dimensional `<|F|>`, and the
near-axis/bulk/edge split, over the whole domain and over `s` in
`[0.1, 0.99]`.

## Shaped tokamak: the polish gain, measured four ways

`input.shaped_tokamak_pressure_polished`, ns = 31, mpol = 5, degree 3,
14 radial spans, SOLVAX Gauss-Newton, 80 iterations, independently certified.

| quantity | before | after | ratio |
|---|---|---|---|
| `eps_F` volume L2 (bounded by 2) | 1.284e-2 | 1.803e-3 | 7.12x |
| `eps_F` Linf | 1.847e-1 | 5.537e-3 | 33.3x |
| dimensional \|F\| volume L2 [N m^-3] | 3.330e2 | 2.068e2 | 1.61x |
| `<\|F\|>` whole domain [N m^-3] | 2.337e2 | 1.773e2 | 1.32x |
| `<\|F\|>/<\|grad p\|>` whole domain | 2.090e-3 | 1.586e-3 | 1.32x |
| `<\|F\|>/<\|grad(B^2/2mu0)\|>` whole domain | 2.374e-3 | 1.801e-3 | 1.32x |
| `<\|F\|>/<\|grad p\|>`, `s` in [0.1, 0.99] | 1.658e-3 | 1.561e-3 | 1.06x |
| \|F\| L2 near axis, `rho < 0.2` [N m^-3] | 9.773e2 | 6.718e1 | 14.5x |
| \|F\| L2 bulk, `0.2 <= rho <= 0.8` [N m^-3] | 1.630e2 | 1.470e2 | 1.11x |
| \|F\| L2 edge, `rho > 0.8` [N m^-3] | 4.089e2 | 2.844e2 | 1.44x |

The polish is not uniformly 7x better. It removes a factor of 14.5 from the
near-axis residual and about 10% from the bulk; the 7.12x in `eps_F` is
mostly that near-axis gain re-expressed in a metric whose denominator is
smallest exactly there. As a volume-averaged relative force error the
improvement is 1.32x over the whole domain and 1.06x over `s` in
`[0.1, 0.99]`. This deck is not saturated — `<|grad p|>` is 1.1e5 Pa m^-1
and `eps_F` is far from 2 — so this is a weighting effect, not a collapsed
denominator.

The README previously quoted "about 26-fold, from 5.05e-2 to 1.91e-3". Those
two numbers are the *exported* wouts of `readme_polish_summary.webp`, written
on different radial meshes (`ns = 31` solve mesh unpolished, `ns = 129`
certifiable mesh polished), so their ratio multiplies the polish gain by an
export-mesh reconstruction difference. It is withdrawn.

## Solovev: the same certificate with its denominator collapsed

`input.solovev`, ns = 15, mpol = 4, degree 3, 8 radial spans,
`--diagnostics-only` — a single-state certificate, not a before/after pair.
(The driver returns the *unpolished* state when a correction fails to
certify, so a before/after pair on a deck that cannot reach the 1e-2 bar
would be two copies of one state. That is why this record is
diagnostics-only rather than a polish run with a raised tolerance.)

The bundled deck peaks at 0.125 Pa. Its pressure gradient is five orders of
magnitude below its magnetic pressure gradient, so the pointwise denominator
`|JxB| + |grad p|` is, to any useful precision, just `|JxB|`:

| quantity | value |
|---|---|
| `eps_F` volume L2 | 1.969 |
| `eps_F` Linf | 2.000 |
| `<\|grad p\|>` [Pa m^-1] | 1.35e-1 |
| `<\|grad(B^2/2mu0)\|>` [Pa m^-1] | 8.03e3 |
| `<\|F\|>` [N m^-3] | 4.00e1 |
| `<\|F\|>/<\|grad p\|>` (Panici) | 2.96e2 |
| `\|F\|` L2 / `<\|grad(B^2/2mu0)\|>` | 1.22e-2 |

`eps_F` is at its ceiling and carries no information: a reader told only
"1.969" would conclude the state is 200% out of force balance. The
vacuum-safe normalization says 1.2e-2, an ordinary residual for a lifted
ns = 15 solve. The Panici ratio is 296 — arithmetically correct, and equally
useless, which is why the record reports it next to the denominator that
produced it rather than on its own.
