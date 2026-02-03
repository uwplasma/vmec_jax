# Gradients

Autodiff and implicit differentiation examples (generally requires JAX).

- `implicit_lambda_gradients.py`: implicit differentiation for the lambda-only solve.
- `implicit_fixed_boundary_sensitivity.py`: implicit differentiation through the fixed-boundary solve.
- `grad_vmec_tomnsps_residual.py`: gradients through the Step-10 VMEC-style residual pipeline.

Tip: enable 64-bit in JAX for parity work (`vmec_jax._compat.enable_x64(True)`).

