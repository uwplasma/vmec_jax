"""Static (compile-time) data for vmec_jax.

Step-1 introduces a small "static" container that groups together data that
should be precomputed once per equilibrium problem and then reused inside
`jax.jit`'d kernels:

- mode table (m,n)
- angle grids (theta,zeta)
- helical basis tensors (cos/sin on the grid)
- radial grid (s in [0,1])

Keeping these pieces together prevents accidental recomputation and makes it
easier to write fast, end-to-end differentiable kernels later.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._compat import jnp
from .config import VMECConfig
from .fourier import HelicalBasis, build_helical_basis
from .grids import AngleGrid, make_angle_grid
from .modes import ModeTable, vmec_mode_table


@dataclass(frozen=True)
class VMECStatic:
    """Precomputed static data for a VMEC run."""

    cfg: VMECConfig
    modes: ModeTable
    grid: AngleGrid
    basis: HelicalBasis
    s: any  # (ns,) radial coordinate in [0,1]


def build_static(cfg: VMECConfig) -> VMECStatic:
    """Build the VMECStatic container from a parsed config."""
    modes = vmec_mode_table(cfg.mpol, cfg.ntor)
    grid = make_angle_grid(cfg.ntheta, cfg.nzeta, cfg.nfp, endpoint=False)
    basis = build_helical_basis(modes, grid)
    # Radial coordinate s = (i)/(ns-1). VMEC uses "s" = normalized toroidal flux.
    # For step-1 we only need a monotone [0,1] grid.
    if cfg.ns < 2:
        s = jnp.asarray([0.0])
    else:
        s = jnp.linspace(0.0, 1.0, cfg.ns)
    return VMECStatic(cfg=cfg, modes=modes, grid=grid, basis=basis, s=s)
