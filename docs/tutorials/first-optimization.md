# Optimize a boundary

In this lesson you reshape the circular tokamak's boundary until its aspect
ratio is exactly 4.0 — three cost evaluations, under a minute on a laptop
CPU.

## The optimization

```python
import vmex as vj
from vmex.core import optimize as opt

inp = vj.VmecInput.from_file("input.circular_tokamak")
result = opt.least_squares(
    [(opt.aspect_ratio, 4.0, 1.0)],   # (function, target, weight)
    inp,
    max_mode=1,                        # free only the m,|n| <= 1 boundary modes
    jac="implicit",                    # exact gradients from the adjoint
    verbose=2,
)
print(float(opt.aspect_ratio(result.equilibrium.state,
                             result.equilibrium.runtime)))
```

The three arguments to understand:

- **Terms** are simsopt-style `(function, target, weight)` triples; the cost
  is `sum(weight * (function - target)**2) / 2`. Any objective from
  {doc}`/reference/objectives` slots in, and several terms combine freely.
- **`max_mode=1`** frees the boundary Fourier coefficients with
  `m, |n| <= 1` (`RBC(0,0)` stays fixed); higher values release finer
  harmonics.
- **`jac="implicit"`** uses the exact implicit-differentiation Jacobian from
  {doc}`first-gradient` instead of finite differences.

## What you see

```text
[least_squares] cost = 5.000000e-01
[least_squares] cost = 1.622593e+32
[least_squares] cost = 3.355398e-30
```

Evaluation 1 is the seed (aspect 3, so `(3-4)^2/2 = 0.5`). Evaluation 2 is a
failed trial — the trust region proposed a boundary whose solve diverged, got
back a large finite cost instead of a crash, and shrank the step. Evaluation
3 lands the answer: cost 3.4e-30, aspect ratio 4.0 to 4e-16.

## The result object

`result` is a scipy `OptimizeResult` augmented with three fields:

- `result.input` — the optimized {class}`~vmex.core.input.VmecInput`; write
  it out with `result.input.to_indata("input.optimized")`;
- `result.equilibrium` — the final converged equilibrium (state, runtime,
  lazy `.wout`), ready for further diagnostics;
- `result.solve_stats` — solve/iteration counts for the whole campaign.

## Where to go from here

A one-term, `max_mode=1` problem is the smallest possible campaign. Real
campaigns combine quasisymmetry or QI residuals with geometric targets,
release all harmonics at once under Exponential Spectral Scaling, and reach
precise QA in one 14.5-minute CPU call — that recipe, with the QA/QH/QI/QP
decks to reproduce it, is {doc}`/howto/optimize-a-boundary`.
