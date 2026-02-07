"""Public, user-facing API for vmec_jax.

This module intentionally re-exports a small set of functions that cover the
common workflows:
- I/O: load input, read/write wout
- Solve: fixed-boundary driver
- Plotting: VMEC-style surfaces and B magnitude

Advanced users can import lower-level kernels directly from submodules.
"""

from __future__ import annotations

from .driver import (
    FixedBoundaryRun,
    load_example,
    load_input,
    load_wout,
    run_fixed_boundary,
    step10_fsq_from_state,
    write_wout_from_fixed_boundary_run,
)
from .plotting import (
    bmag_from_state_physical,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    surface_rz_from_wout_physical,
    surface_stack,
    zeta_grid,
    zeta_grid_field_period,
)
from .wout import read_wout, state_from_wout
from .workflows import (
    axisym_showcase,
    export_vtk_surface_and_fieldline,
    step10_getfsq_parity_cases,
    write_axisym_overview,
    write_bmag_parity_figures,
    write_bsub_parity_figures,
    write_bsup_parity_figures,
)

__all__ = [
    # Driver / solve
    "FixedBoundaryRun",
    "load_example",
    "load_input",
    "load_wout",
    "run_fixed_boundary",
    "step10_fsq_from_state",
    "write_wout_from_fixed_boundary_run",
    # Plotting helpers
    "bmag_from_state_physical",
    "closed_theta_grid",
    "fix_matplotlib_3d",
    "profiles_from_wout",
    "surface_rz_from_wout_physical",
    "surface_stack",
    "zeta_grid",
    "zeta_grid_field_period",
    # Low-friction wout access for plotting
    "read_wout",
    "state_from_wout",
    # Workflows (used by examples)
    "axisym_showcase",
    "write_axisym_overview",
    "write_bsup_parity_figures",
    "write_bsub_parity_figures",
    "write_bmag_parity_figures",
    "export_vtk_surface_and_fieldline",
    "step10_getfsq_parity_cases",
]
