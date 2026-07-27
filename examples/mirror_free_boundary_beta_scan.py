"""Solved two-coil free-boundary mirror beta scan and physics plots.

Run from the repository root with::

    python examples/mirror_free_boundary_beta_scan.py

The first four points are the supported 0--10% validation scan. The last two
are continuation states whose independent force/refinement gates do not pass.
Every curve and surface comes from a coupled
plasma-boundary-vacuum equilibrium solve with residual tolerance ``FTOL``; no
prescribed finite-beta boundary is plotted.
"""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmex.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    SplineMirrorDiscretization,
    mout_from_result,
    plot_mout,
    solve_beta_scan,
    write_mout,
)
from vmex.mirror.output import (  # noqa: E402
    plot_axisymmetric_beta_scan_summary,
    summarize_axisymmetric_beta_scan,
)
from vmex.mirror.output import (  # noqa: E402
    FreeBoundaryRestart,
    load_free_boundary_restart,
    save_free_boundary_restart,
)

# Inputs: edit these values, then run the file directly.
BETAS = np.asarray([0.0, 0.01, 0.03, 0.10, 0.25, 0.50])
SUPPORTED_BETA_MAX = 0.10
STRONG_FORCE_GATE = 5.0e-2
NS = 7
NXI = 13
SPLINE_ELEMENTS = 7
EXTERIOR_NTHETA = 12
EXTERIOR_ORDER = 6
EXTERIOR_SPECTRAL_SIDE_DENSITY = True
FTOL = 1.0e-12
MAX_ITERATIONS = 2000
Z_MIN, Z_MAX = -0.8, 0.8
# Compact coils sized to the plasma: same vacuum on-axis midplane field as the
# former 0.9 m / 2.0e5 A loops (B(0) ~ 0.0836 T), with a deeper mirror well.
COIL_RADIUS = 0.5
COIL_SEPARATION = 2.0
COIL_CURRENT = 3.72e5
CENTER_RADIUS = 0.25
OUTPUT_DIR = Path("results/mirror_free_boundary_beta_scan")
SAVE_RESTARTS = True
RESTART_FROM = None  # e.g. OUTPUT_DIR / "beta_003p0pct.npz"; then trim BETAS

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

coil_dofs = np.zeros((2, 3, 3))
coil_dofs[:, 0, 2] = COIL_RADIUS
coil_dofs[:, 1, 1] = COIL_RADIUS
coil_dofs[:, 2, 0] = np.asarray([-0.5, 0.5]) * COIL_SEPARATION
try:
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart
except ModuleNotFoundError as error:
    raise ModuleNotFoundError("This example requires ESSOS: pip install -e /path/to/ESSOS") from error

coils = Coils(
    Curves(jnp.asarray(coil_dofs), n_segments=128, nfp=1, stellsym=False),
    jnp.full(2, COIL_CURRENT),
)
biot_savart = BiotSavart(coils)


def external_field(points):
    """Evaluate the ESSOS field on an arbitrary array of Cartesian points."""

    points = jnp.asarray(points)
    return jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)


config = MirrorConfig(
    resolution=MirrorResolution(ns=NS, mpol=0, nxi=NXI),
    z_min=Z_MIN,
    z_max=Z_MAX,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
source_grid = config.build_grid()
discretization = SplineMirrorDiscretization.build_cgl(config, elements=SPLINE_ELEMENTS)
grid = discretization.grid
initial_restart = None if RESTART_FROM is None else load_free_boundary_restart(RESTART_FROM, discretization)
z = jnp.asarray(grid.z)
coil_z = 0.5 * COIL_SEPARATION
vacuum_axis_field = sum(
    4.0e-7 * jnp.pi * COIL_CURRENT * COIL_RADIUS**2 / (2.0 * (COIL_RADIUS**2 + (z - position) ** 2) ** 1.5)
    for position in (-coil_z, coil_z)
)
center = int(np.argmin(np.abs(grid.z)))
axial_flux_derivative = 0.5 * vacuum_axis_field[center] * CENTER_RADIUS**2
initial_boundary = discretization.fit_boundary(
    MirrorBoundary.from_axis_field(
        axial_flux_derivative,
        vacuum_axis_field,
        grid,
    ),
    source_grid,
)
print(f"Solving {BETAS.size} beta points at ns={NS}, nxi={NXI}, ftol={FTOL:.0e}")
results = solve_beta_scan(
    initial_boundary,
    discretization,
    config,
    external_field,
    jnp.asarray(BETAS),
    axial_flux_derivative=axial_flux_derivative,
    reference_field=float(vacuum_axis_field[center]),
    initial_restart=initial_restart,
    exterior_ntheta=EXTERIOR_NTHETA,
    exterior_order=EXTERIOR_ORDER,
    exterior_spectral_side_density=EXTERIOR_SPECTRAL_SIDE_DENSITY,
)
gamma = np.asarray(coils.gamma)
if SAVE_RESTARTS:
    for beta, result in zip(BETAS, results, strict=True):
        label = f"beta_{100 * beta:05.1f}pct".replace(".", "p")
        save_free_boundary_restart(OUTPUT_DIR / label, FreeBoundaryRestart.from_result(result))
for beta, result in zip(BETAS, results, strict=True):
    label = f"beta_{100 * beta:05.1f}pct".replace(".", "p")
    write_mout(
        OUTPUT_DIR / f"mout_mirror_{label}.nc",
        mout_from_result(
            result,
            grid,
            config,
            axial_flux_derivative=axial_flux_derivative,
            coil_xyz=gamma,
        ),
    )
diagnostics = summarize_axisymmetric_beta_scan(
    results,
    jnp.asarray(BETAS),
    grid,
    reference_field=float(vacuum_axis_field[center]),
)


summary = [
    {key: float(value) for key, value in vars(item).items()}
    | {
        "variational_max": float(result.variational_max),
        "pointwise_force_rms": float(result.plasma_force.normalized_rms),
        "supported_lane": bool(
            item.requested_beta <= SUPPORTED_BETA_MAX
            and float(result.plasma_force.normalized_rms) < STRONG_FORCE_GATE
        ),
        "model_supported_beta_range": bool(item.requested_beta <= SUPPORTED_BETA_MAX),
        "passes_strong_force_gate": bool(
            float(result.plasma_force.normalized_rms) < STRONG_FORCE_GATE
        ),
    }
    for item, result in zip(diagnostics, results, strict=True)
]
(OUTPUT_DIR / "beta_scan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for row in summary:
    if row["model_supported_beta_range"]:
        assert row["passes_strong_force_gate"], f"supported beta point failed the force gate: {row}"

middle_beta = min(0.10, 0.5 * float(BETAS[-1]))
display_indices = sorted({0, int(np.argmin(np.abs(BETAS - middle_beta))), len(BETAS) - 1})
for index in display_indices:
    label = f"beta_{100 * BETAS[index]:05.1f}pct".replace(".", "p")
    plot_mout(
        OUTPUT_DIR / f"mout_mirror_{label}.nc",
        OUTPUT_DIR,
        name=f"mirror_{label}",
    )

mirror_ratio = float(jnp.max(vacuum_axis_field) / vacuum_axis_field[center])
radius_expansion = 100.0 * (summary[-1]["center_radius"] / summary[0]["center_radius"] - 1.0)
field_reduction = 100.0 * (1.0 - summary[-1]["diamagnetic_field_ratio"])
final_gate = (
    "its independent force gate fails"
    if not summary[-1]["passes_strong_force_gate"]
    else "beyond the supported model range"
)
caption = (
    f"Two ESSOS loops (radius {COIL_RADIUS} m at z = +/-{0.5 * COIL_SEPARATION} m, "
    f"{COIL_CURRENT:.3g} A each) give vacuum B(0) = {float(vacuum_axis_field[center]):.4f} T and "
    f"on-grid mirror ratio {mirror_ratio:.2f}. Betas through {100 * SUPPORTED_BETA_MAX:g}% pass the "
    f"strong-force gate; the {100 * float(BETAS[-1]):g}% validation continuation expands the center radius by "
    f"{radius_expansion:.2f}% and lowers the on-axis field by {field_reduction:.2f}% ({final_gate})."
)
composite = plot_axisymmetric_beta_scan_summary(
    [
        (
            f"beta = {100 * beta:g}%",
            OUTPUT_DIR / f"mout_mirror_{f'beta_{100 * beta:05.1f}pct'.replace('.', 'p')}.nc",
            bool(row["supported_lane"]),
        )
        for beta, row in zip(BETAS, summary, strict=True)
    ],
    OUTPUT_DIR,
    display=tuple(display_indices),
    strong_force_gate=STRONG_FORCE_GATE,
)

# The figure stays clean (short title + panel labels only); the coil geometry,
# vacuum field, mirror ratio, and beta observables are reported here and in
# docs/mirror_geometry.rst.
print(caption)
print(json.dumps(summary, indent=2))
print(f"Wrote solved-state 3D, cross-section, |B|, and summary plots in {OUTPUT_DIR}")
print(f"Wrote beta-scan composite figure: {composite}")
