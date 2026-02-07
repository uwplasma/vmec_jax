"""vmec_jax: a JAX port of VMEC2000 (work in progress).

Step-0 contains:
- VMEC &INDATA parser
- mode table + angle grids
- helical Fourier basis evaluation
- boundary evaluation
- state packing/unpacking (radial x modes)

The full equilibrium solver will be built incrementally in later steps.
"""

from . import api
from .namelist import read_indata, InData
from .config import VMECConfig, load_config
from .modes import ModeTable, vmec_mode_table, nyquist_mode_table, default_grid_sizes
from .grids import AngleGrid, make_angle_grid
from .boundary import BoundaryCoeffs, boundary_from_indata
from .fourier import (
    HelicalBasis,
    build_helical_basis,
    eval_fourier,
    eval_fourier_dtheta,
    eval_fourier_dzeta_phys,
)
from .plotting import (
    SurfaceData,
    axis_rz_from_wout,
    axis_rz_from_wout_physical,
    axis_rz_from_state_physical,
    bmag_from_wout,
    bmag_from_wout_physical,
    bmag_from_state_physical,
    bmag_from_state_vmec_realspace,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    select_zeta_slices,
    surface_data_from_wout,
    surface_rz_from_wout,
    surface_rz_from_wout_physical,
    surface_rz_from_state,
    surface_rz_from_state_physical,
    surface_stack,
    vmecplot2_cross_section_indices,
    zeta_grid,
    zeta_grid_field_period,
)
from .driver import (
    ExampleData,
    FixedBoundaryRun,
    example_paths,
    load_example,
    load_input,
    load_wout,
    run_fixed_boundary,
    save_npz,
    step10_fsq_from_state,
    write_wout_from_fixed_boundary_run,
)
from .state import VMECState, pack_state, unpack_state
from .static import VMECStatic, build_static
from .init_guess import initial_guess_from_boundary
from .coords import Coords, eval_coords
from .geom import Geom, eval_geom
from .profiles import ProfileInputs, profiles_from_indata, eval_profiles
from .integrals import dvds_from_sqrtg, cumtrapz_s, volume_from_sqrtg
from .field import bsup_from_geom, bsup_from_sqrtg_lambda, b2_from_bsup
from .energy import magnetic_wb_from_state
from .diagnostics import Summary, print_jacobian_stats, print_summary, summarize_array, summarize_many
from .vmec_jacobian import vmec_half_mesh_jacobian_from_state
from .vmec_realspace import vmec_realspace_analysis, vmec_realspace_geom_from_state
from .implicit import (
    ImplicitFixedBoundaryOptions,
    ImplicitLambdaOptions,
    solve_fixed_boundary_state_implicit,
    solve_lambda_state_implicit,
)
from .solve import (
    SolveFixedBoundaryResult,
    SolveLambdaResult,
    solve_fixed_boundary_gd,
    solve_fixed_boundary_lbfgs,
    solve_fixed_boundary_lbfgs_vmec_residual,
    solve_fixed_boundary_gn_vmec_residual,
    solve_fixed_boundary_vmecpp_iter,
    solve_lambda_gd,
)
from .residuals import ForceResiduals, force_residuals_from_state

__all__ = [
    "api",
    "read_indata",
    "InData",
    "VMECConfig",
    "load_config",
    "ModeTable",
    "vmec_mode_table",
    "nyquist_mode_table",
    "default_grid_sizes",
    "AngleGrid",
    "make_angle_grid",
    "BoundaryCoeffs",
    "boundary_from_indata",
    "HelicalBasis",
    "build_helical_basis",
    "eval_fourier",
    "eval_fourier_dtheta",
    "eval_fourier_dzeta_phys",
    "SurfaceData",
    "axis_rz_from_wout",
    "axis_rz_from_wout_physical",
    "axis_rz_from_state_physical",
    "bmag_from_wout",
    "bmag_from_wout_physical",
    "bmag_from_state_physical",
    "bmag_from_state_vmec_realspace",
    "closed_theta_grid",
    "fix_matplotlib_3d",
    "profiles_from_wout",
    "select_zeta_slices",
    "surface_data_from_wout",
    "surface_rz_from_wout",
    "surface_rz_from_wout_physical",
    "surface_rz_from_state",
    "surface_rz_from_state_physical",
    "surface_stack",
    "vmecplot2_cross_section_indices",
    "zeta_grid",
    "zeta_grid_field_period",
    "ExampleData",
    "FixedBoundaryRun",
    "example_paths",
    "load_example",
    "load_input",
    "load_wout",
    "run_fixed_boundary",
    "save_npz",
    "step10_fsq_from_state",
    "write_wout_from_fixed_boundary_run",
    "VMECState",
    "pack_state",
    "unpack_state",
    "VMECStatic",
    "build_static",
    "initial_guess_from_boundary",
    "Coords",
    "eval_coords",
    "Geom",
    "eval_geom",
    "ProfileInputs",
    "profiles_from_indata",
    "eval_profiles",
    "dvds_from_sqrtg",
    "cumtrapz_s",
    "volume_from_sqrtg",
    "bsup_from_geom",
    "bsup_from_sqrtg_lambda",
    "b2_from_bsup",
    "magnetic_wb_from_state",
    "Summary",
    "print_jacobian_stats",
    "print_summary",
    "summarize_array",
    "summarize_many",
    "vmec_half_mesh_jacobian_from_state",
    "vmec_realspace_analysis",
    "vmec_realspace_geom_from_state",
    "ImplicitLambdaOptions",
    "ImplicitFixedBoundaryOptions",
    "solve_lambda_state_implicit",
    "solve_fixed_boundary_state_implicit",
    "SolveLambdaResult",
    "solve_lambda_gd",
    "SolveFixedBoundaryResult",
    "solve_fixed_boundary_gd",
    "solve_fixed_boundary_lbfgs",
    "solve_fixed_boundary_lbfgs_vmec_residual",
    "solve_fixed_boundary_gn_vmec_residual",
    "solve_fixed_boundary_vmecpp_iter",
    "ForceResiduals",
    "force_residuals_from_state",
]
