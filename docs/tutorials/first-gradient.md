# Differentiate an equilibrium

In this lesson you compute an exact derivative of a converged equilibrium —
`d(aspect ratio)/d(boundary coefficient)` — with `jax.grad`, and check it
against finite differences.

## The three lines that matter

```python
import jax
from vmex.core import implicit
from vmex.core.input import VmecInput

inp = VmecInput.from_file("input.solovev")
p0 = implicit.params_from_input(inp)          # differentiable parameters
grad = jax.grad(lambda p: implicit.run(inp, p).aspect)(p0)
```

{func}`vmex.core.implicit.run` is the differentiable solve entry point: it
converges the equilibrium and returns a pytree whose scalar outputs
(`aspect`, `wb`, `volume`, ...) can be wrapped in any JAX transformation.
`grad` is a pytree matching `p0`: one derivative per boundary Fourier
coefficient, profile coefficient, `phiedge`, `pres_scale`, `curtor`.

## Check it

`examples/take_gradients.py` runs the check on the bundled Solovev deck
(ns=11, ftol=1e-12), comparing the adjoint gradient against a central finite
difference through two full re-solves:

```text
d(aspect)/d(RBC(0,1))  AD=-1.5182532273e+00  FD=-1.5182532303e+00  rel=2.00e-09
d(wb)/d(phiedge)       AD=+1.2910254037e-01  FD=+1.2910254037e-01  rel=1.97e-12
```

The agreement is at the finite-difference noise floor — the adjoint side has
no step size at all.

```{literalinclude} ../../examples/take_gradients.py
:language: python
```

## Why there is no step size to tune

The gradient does not difference two solves and does not backpropagate
through the iteration. It applies the implicit function theorem at the
converged fixed point: one linear (adjoint) solve per scalar output, O(1)
memory no matter how many iterations the forward solve took
({doc}`/explanation/adjoint-gradients`). Two consequences you get for free:

- cost is independent of the number of parameters — the gradient above has
  one entry per boundary coefficient at the price of one adjoint solve;
- there is no finite-difference truncation/roundoff trade-off, which is
  exactly what the 2e-9 agreement shows.

One caveat worth knowing before you rely on FD checks yourself: for
solver-sensitive outputs (iota, mirror ratio, magnetic well) a naive
re-solving finite difference is not a valid reference — the frozen-path
check in {doc}`/explanation/adjoint-gradients` is.

Next: {doc}`first-optimization` puts the gradient to work.
