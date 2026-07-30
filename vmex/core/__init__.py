"""Clean-room core of vmex (§5) — will replace the legacy modules.

Module map (each header docstring names its VMEC2000 counterpart):

- ``errors``          typed zero-crash exceptions + werror table
- ``printing``        VMEC2000-format console output (byte-exact)
- ``input``           VmecInput: INDATA + VMEC++-JSON parsing, round-trip writers
- ``profiles``        pressure/iota/current parameterizations (pure jnp)
- ``fourier``         Resolution, ModeTable, trig tables (fixaray.f)
- ``transforms``      totzsps/totzspa/tomnsps/tomnspa as batched matmuls
- ``geometry``        real-space R/Z/lambda, half-mesh jacobian (jacobian.f)
- ``fields``          metrics, B components, energies, tcon (bcovar.f)
- ``forces``          MHD force kernels + spectral condensation (forces.f, alias.f)
- ``residuals``       m=1 constraint, fsqr/fsqz/fsql, preconditioned lane (residue.f90)
- ``preconditioner``  1D radial tridiagonal preconditioner (precondn.f, scalfor.f)
- ``step``            Richardson stepping + restart control (evolve.f, restart.f)
- ``setup``           radial profiles + initial guess (profil1d/3d.f, readin.f)
- ``solver``          single-grid fixed-boundary solve loop (funct3d.f, eqsolve.f)
- ``statephysics``    shared state-physics primitives (_field_chain, half-mesh iota/sampling)
- ``implicit``        implicit differentiation of the equilibrium (custom VJP + adjoint GMRES)
- ``stability``       differentiable ideal-MHD stability (Mercier and infinite-n ballooning)
- ``freeboundary_diff`` differentiable free-boundary residual via virtual casing (R15.3/R19)
- ``device``          CPU/GPU placement policy (measured: benchmarks/gpu_baseline.json)

Every module is validated by A/B equivalence tests against the legacy
parity-proven implementation in ``tests/``; the solve loop is
validated end-to-end against VMEC2000 golden runs
(``tests/test_solver_end_to_end.py``).
"""
