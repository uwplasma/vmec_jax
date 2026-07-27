#!/usr/bin/env python
"""QI-mirror hybrid: Fourier (VMEC-native) vs B-spline axis representations.

Physics.  A quasi-isodynamic (QI) stellarator has poloidally closed ``|B|``
contours and near-straight (low-curvature) magnetic-axis segments at its
field-period-symmetric planes.  Cutting the axis there and inserting a straight
magnetic-mirror cell is therefore natural: the seam sits where the QI axis is
already almost straight.  An ``nfp = 2`` QI axis has two such planes per field
period, so it is cut at all *four* low-curvature planes, and at each cut a
straight leg is inserted **along the local axis tangent** (extending the axis in
its own direction, not a shared transverse direction).  Choosing the leg lengths
so the inserted displacements cancel and reflecting one half about the symmetry
axis makes the four-legged racetrack stellarator symmetric.  This example builds
that QI-mirror hybrid and compares the two ways of representing the spliced axis.

  * Fourier (VMEC-native) is a *global* basis: a truncated series rings at the
    straight->curved seam (a Gibbs-type feature) and its error decays only
    ~1/N, everywhere at once.
  * B-splines (the ``vmex.mirror`` closed-spline lane) are *local*: the straight
    mirror legs are reproduced to machine precision on their interior, with
    error confined to a fixed number of knot-spacings around the junction.

The B-spline lane also *solves* the hybrid equilibrium (divergence-free field,
force residual, rotational transform), reusing the same machinery as the
racetrack hybrid.  The QI reference equilibrium (iota, ``|B|``) is the VMEC
Fourier solve of ``input.nfp2_QI``.

Honesty.  The B-spline mirror equilibrium is a scalar-pressure spline model, not
a VMEC re-solve; its transform comes from a weak axial current, so it is *not* a
reproduction of the QI transform -- it demonstrates the geometry and the
exactly-straight mirror cell.  A literal VMEC re-solve of a straight-axis device
is degenerate in cylindrical ``(R, phi, Z)`` coordinates (a straight axis
segment cannot be parameterised by the cylindrical angle), which is precisely
why the closed-axis B-spline lane exists.  See ``docs/mirror_geometry.rst``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import vmex as vj
from vmex.core.plotting import axis_rz
from vmex.mirror import (
    MirrorConfig,
    MirrorResolution,
    build_qi_mirror_hybrid,
    solve_fixed_boundary,
    splice_straight_legs,
    trace_closed_field_line,
)
from vmex.mirror.basis import CubicBSplineBasis
from vmex.mirror.geometry import magnetic_field_squared

# --------------------------- parameters ------------------------------------
INPUT_FILE = REPO_ROOT / "examples" / "data" / "input.nfp2_QI"
FIGURE_PATH = REPO_ROOT / "docs" / "_static" / "figures" / "qi_mirror_hybrid.png"
OUTPUT_DIR = Path("results/qi_mirror_hybrid")

N_AXIS = 512                    # magnetic-axis samples over the full torus
STRAIGHT_LENGTH = 1.2           # inserted mirror-leg length [m]
SECTION_RADIUS = 0.12           # circular hybrid cross-section radius [m]
COEFFICIENT_COUNT = 64          # B-spline controls in the solved hybrid axis
FOURIER_MODES = (8, 16, 32, 64)         # toroidal harmonics N in the sweep
BSPLINE_CONTROLS = (32, 64, 128, 256)   # closed-spline controls M in the sweep
MIRROR_NS, MIRROR_MPOL, MIRROR_NXI = 5, 3, 4
MIRROR_FTOL = 1.0e-11
MIRROR_MAX_ITER = 800
AXIAL_FLUX_DERIVATIVE = 0.02
CURRENT_DERIVATIVE = 0.002

CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"  # reduced smoke budget
if CI:
    N_AXIS = 128
    COEFFICIENT_COUNT = 16
    FOURIER_MODES = (8, 16)
    BSPLINE_CONTROLS = (16, 32)
    MIRROR_FTOL, MIRROR_MAX_ITER = 1.0e-6, 40


# ------------------------- QI axis + curvature ------------------------------
def qi_axis_and_curvature(wout, nfp: int, n_axis: int):
    """Closed magnetic-axis curve r(phi) and its 3-D curvature kappa(phi)."""

    phi = np.linspace(0.0, 2.0 * np.pi, n_axis, endpoint=False)
    r_axis, z_axis = axis_rz(wout, phi)
    points = np.stack([r_axis * np.cos(phi), r_axis * np.sin(phi), z_axis], axis=1)
    wavenumber = np.fft.fftfreq(n_axis, d=phi[1] - phi[0]) * 2.0 * np.pi
    spectrum = np.fft.fft(points, axis=0)
    first = np.real(np.fft.ifft(1j * wavenumber[:, None] * spectrum, axis=0))
    second = np.real(np.fft.ifft(-(wavenumber[:, None] ** 2) * spectrum, axis=0))
    speed = np.linalg.norm(first, axis=1)
    curvature = np.linalg.norm(np.cross(first, second), axis=1) / speed**3
    return phi, points, curvature


def symmetric_cut_indices(curvature: np.ndarray, nfp: int) -> tuple[int, ...]:
    """The ``2 * nfp`` low-curvature symmetry planes of a QI axis.

    A stellarator-symmetric QI axis is nearly straight at two planes per field
    period, so the curvature has ``2 * nfp`` minima (four for ``nfp = 2``).  All
    of them are cut so the straight mirror legs inserted along the local axis
    tangent are arranged stellarator symmetrically.
    """

    prev, nxt = np.roll(curvature, 1), np.roll(curvature, -1)
    minima = np.where((curvature < prev) & (curvature <= nxt))[0]
    if minima.size < 2 * nfp:
        raise ValueError(f"expected >= {2 * nfp} curvature minima, found {minima.size}")
    deepest = minima[np.argsort(curvature[minima])[: 2 * nfp]]
    return tuple(sorted(int(i) for i in deepest))


# ------------------------- representation fits ------------------------------
def fourier_fit(parameter: np.ndarray, curve: np.ndarray, modes: int) -> np.ndarray:
    """Least-squares truncated Fourier series of a periodic curve (global)."""

    columns = [np.ones_like(parameter)]
    for order in range(1, modes + 1):
        columns += [np.cos(order * parameter), np.sin(order * parameter)]
    design = np.stack(columns, axis=1)
    coefficients, *_ = np.linalg.lstsq(design, curve, rcond=None)
    return design @ coefficients


def bspline_fit(splice, controls: int, parameter: np.ndarray) -> np.ndarray:
    """Closed cubic B-spline interpolation at uniform arc length (local)."""

    from vmex.mirror.splines import _sample_closed_polyline

    basis = CubicBSplineBasis.periodic_uniform(controls)
    nodes = np.asarray(basis.collocation_nodes)
    control_values = _sample_closed_polyline(
        splice.points, nodes / (2.0 * np.pi) * splice.total_length
    )
    coefficients = basis.fit(control_values, axis=0)
    return np.asarray(basis.evaluate(coefficients, parameter, axis=0))


def bspline_leg_midpoint_error(splice, controls: int) -> float:
    """Deviation of the spline from the exact straight leg at its midpoint."""

    from vmex.mirror.splines import _sample_closed_polyline

    basis = CubicBSplineBasis.periodic_uniform(controls)
    nodes = np.asarray(basis.collocation_nodes)
    coefficients = basis.fit(
        _sample_closed_polyline(splice.points, nodes / (2.0 * np.pi) * splice.total_length),
        axis=0,
    )
    start, stop = splice.leg_windows[0]
    midpoint = 0.5 * (start + stop)
    fitted = np.asarray(
        basis.evaluate(coefficients, np.array([midpoint / splice.total_length * 2.0 * np.pi]), axis=0)
    )[0]
    exact = _sample_closed_polyline(splice.points, np.array([midpoint]))[0]
    return float(np.linalg.norm(fitted - exact))


def leg_interior_mask(arc_length: np.ndarray, splice, margin_fraction: float) -> np.ndarray:
    """Points inside either straight leg, away from the junctions by a margin."""

    margin = margin_fraction * STRAIGHT_LENGTH
    mask = np.zeros(arc_length.shape, dtype=bool)
    for start, stop in splice.leg_windows:
        mask |= (arc_length >= start + margin) & (arc_length <= stop - margin)
    return mask


def main() -> None:  # noqa: PLR0915 - a single linear example script
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- QI reference equilibrium (Fourier / VMEC) ----------------
    inp = vj.VmecInput.from_file(INPUT_FILE)
    if CI:
        inp = dataclasses.replace(inp, ns_array=[13], ftol_array=[1e-8], niter_array=[2000])
    else:
        inp = dataclasses.replace(inp, ns_array=[15, 31], ftol_array=[1e-9, 1e-11],
                                  niter_array=[3000, 6000])
    print(f"QI reference: {INPUT_FILE.name}  nfp={inp.nfp} mpol={inp.mpol} ntor={inp.ntor}")
    qi = vj.solve_multigrid(inp, verbose=False)
    wout = vj.wout_from_state(inp=inp, state=qi.state, fsqr=float(qi.fsqr),
                              fsqz=float(qi.fsqz), fsql=float(qi.fsql),
                              niter=int(qi.iterations), converged=bool(qi.converged))
    iota = np.asarray(wout.iotaf)
    qi_iota_edge, qi_iota_axis = float(iota[-1]), float(iota[0])
    print(f"  converged={bool(qi.converged)}  |iota| axis={abs(qi_iota_axis):.3f} "
          f"edge={abs(qi_iota_edge):.3f}  B0={float(wout.b0):.3f}  aspect={float(wout.aspect):.2f}")

    # ---------------- QI axis, curvature, and cut locations --------------------
    phi, axis_points, curvature = qi_axis_and_curvature(wout, int(inp.nfp), N_AXIS)
    cut = symmetric_cut_indices(curvature, int(inp.nfp))
    print(f"\naxis curvature: min={curvature.min():.3e} max={curvature.max():.3e} 1/m "
          f"(ratio {curvature.max() / curvature.min():.0f}x)")
    print(f"cut locations ({len(cut)} curvature minima): "
          f"phi={[round(float(phi[c]), 3) for c in cut]} rad; "
          f"kappa={[float(f'{curvature[c]:.2e}') for c in cut]} 1/m")

    # ---------------- cut-and-splice the straight mirror legs ------------------
    splice = splice_straight_legs(axis_points, cut_indices=cut, straight_length=STRAIGHT_LENGTH)
    print(f"\nspliced hybrid axis: length={splice.total_length:.3f} m  "
          f"legs={len(splice.leg_windows)}  leg lengths={np.round(splice.leg_lengths, 3).tolist()} m  "
          f"closure={splice.closure_error:.1e}  leg-return corner={splice.corner_angle:.2f} deg")

    # ---------------- representation accuracy: Fourier vs B-spline -------------
    dense = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)
    dense_arc = dense / (2.0 * np.pi) * splice.total_length
    from vmex.mirror.splines import _sample_closed_polyline
    target = _sample_closed_polyline(splice.points, dense_arc)
    interior = leg_interior_mask(dense_arc, splice, margin_fraction=0.20)

    print("\n--- axis representation accuracy (closed QI-mirror hybrid) ---")
    print(f"{'basis':>10} {'DOF':>5} {'rms [m]':>11} {'max [m]':>11} {'leg [m]':>11}")
    fourier_rows, bspline_rows = [], []
    for modes in FOURIER_MODES:
        fit = np.stack([fourier_fit(dense, target[:, j], modes) for j in range(3)], axis=1)
        error = np.linalg.norm(fit - target, axis=1)
        dof = 3 * (2 * modes + 1)
        leg = float(error[interior].max())
        fourier_rows.append((modes, dof, float(np.sqrt((error**2).mean())), float(error.max()), leg))
        print(f"{'Fourier':>10} {dof:5d} {fourier_rows[-1][2]:11.3e} "
              f"{fourier_rows[-1][3]:11.3e} {leg:11.3e}")
    for controls in BSPLINE_CONTROLS:
        fit = bspline_fit(splice, controls, dense)
        error = np.linalg.norm(fit - target, axis=1)
        mid = bspline_leg_midpoint_error(splice, controls)
        bspline_rows.append((controls, 3 * controls, float(np.sqrt((error**2).mean())),
                             float(error.max()), mid))
        print(f"{'B-spline':>10} {3 * controls:5d} {bspline_rows[-1][2]:11.3e} "
              f"{bspline_rows[-1][3]:11.3e} {mid:11.3e}  (leg midpoint)")

    # ---------------- B-spline mirror equilibrium solve ------------------------
    resolution = MirrorResolution(ns=MIRROR_NS, mpol=MIRROR_MPOL, nxi=MIRROR_NXI)
    config = MirrorConfig(resolution=resolution, ftol=MIRROR_FTOL, max_iterations=MIRROR_MAX_ITER)
    setup = build_qi_mirror_hybrid(
        axis_points, resolution, cut_indices=cut, straight_length=STRAIGHT_LENGTH,
        section_radius=SECTION_RADIUS, coefficient_count=COEFFICIENT_COUNT,
        axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
    )
    result = solve_fixed_boundary(
        setup.initial_state, setup.boundary, setup.discretization, config,
        axial_flux_derivative=AXIAL_FLUX_DERIVATIVE, current_derivative=CURRENT_DERIVATIVE,
        solve_lambda=True, axis=setup.axis, require_convergence=False,
    )
    evaluated = result.evaluated
    field_line = trace_closed_field_line(
        evaluated.energy.field, setup.discretization, radial_index=MIRROR_NS - 2, turns=2
    )
    mod_b = np.sqrt(np.maximum(np.asarray(magnetic_field_squared(
        evaluated.energy.field, evaluated.energy.geometry)), 0.0))
    b_axis = float(mod_b[0].mean())
    b_lcfs_min, b_lcfs_max = float(mod_b[-1].min()), float(mod_b[-1].max())
    print("\n--- B-spline mirror-hybrid equilibrium ---")
    print(f"converged={bool(evaluated.converged)}  iterations={int(evaluated.iterations)}")
    print(f"  force normalized rms = {float(evaluated.force.normalized_rms):.3e}")
    print(f"  divergence rms       = {float(evaluated.normalized_divergence_rms):.3e}")
    print(f"  rotational transform = {float(field_line.iota):.4f}")
    print(f"  |B| axis={b_axis:.3f}  LCFS in [{b_lcfs_min:.3f}, {b_lcfs_max:.3f}]  "
          f"mirror ratio={b_lcfs_max / b_lcfs_min:.2f}")

    summary = {
        "qi_iota_axis": qi_iota_axis, "qi_iota_edge": qi_iota_edge,
        "qi_b0": float(wout.b0), "qi_aspect": float(wout.aspect),
        "curvature_min": float(curvature.min()), "curvature_max": float(curvature.max()),
        "cut_phi": [float(phi[c]) for c in cut],
        "cut_kappa": [float(curvature[c]) for c in cut],
        "leg_lengths": [float(x) for x in splice.leg_lengths],
        "splice_length": splice.total_length, "splice_closure": splice.closure_error,
        "corner_angle_deg": splice.corner_angle,
        "fourier": [dict(zip(("modes", "dof", "rms", "max", "leg"), row)) for row in fourier_rows],
        "bspline": [dict(zip(("controls", "dof", "rms", "max", "leg_midpoint"), row)) for row in bspline_rows],
        "hybrid_converged": bool(evaluated.converged),
        "hybrid_iterations": int(evaluated.iterations),
        "hybrid_force_normalized_rms": float(evaluated.force.normalized_rms),
        "hybrid_divergence_rms": float(evaluated.normalized_divergence_rms),
        "hybrid_iota": float(field_line.iota),
        "hybrid_b_axis": b_axis, "hybrid_b_lcfs": [b_lcfs_min, b_lcfs_max],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure_path = FIGURE_PATH if not CI else OUTPUT_DIR / "qi_mirror_hybrid.png"
    make_figure(figure_path, phi, axis_points, curvature, cut, splice, target, dense_arc,
                interior, fourier_rows, bspline_rows, setup, mod_b, summary)
    print(f"\nwrote {figure_path}")
    print(f"wrote {OUTPUT_DIR / 'summary.json'}")


def make_figure(path, phi, axis_points, curvature, cut, splice, target, dense_arc,
                interior, fourier_rows, bspline_rows, setup, mod_b, summary) -> None:
    """House-style comparison figure (jet |B|)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid_c = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    blue, orange = "#2a78d6", "#eda100"
    matplotlib.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": ink, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink2,
        "axes.linewidth": 0.8, "grid.color": grid_c, "grid.linewidth": 0.8,
        "xtick.color": muted, "ytick.color": ink2, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 8.5,
    })

    fig = plt.figure(figsize=(13.6, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # (0,0) QI axis coloured by curvature, cut planes marked ------------------
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    pts = np.concatenate([axis_points, axis_points[:1]], axis=0)
    kap = np.concatenate([curvature, curvature[:1]])
    norm_k = plt.Normalize(curvature.min(), curvature.max())
    for i in range(len(pts) - 1):
        ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], pts[i:i + 2, 2],
                color=plt.cm.jet(norm_k(kap[i])), lw=2.4)
    for c in cut:
        ax.scatter(*axis_points[c], color=ink, s=55, marker="o", depthshade=False, zorder=6)
    ax.set_title("QI magnetic axis, curvature $\\kappa(\\phi)$\ncut at 4 symmetry planes (dots)",
                 fontsize=11, color=ink)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_k, cmap="jet"), ax=ax, shrink=0.55,
                 pad=0.11, label="$\\kappa$ [1/m]")

    # (0,1) spliced QI-mirror hybrid axis: legs (blue) + returns (grey) --------
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    sp = np.concatenate([splice.points, splice.points[:1]], axis=0)
    seg_arc = np.linspace(0.0, splice.total_length, len(sp))
    on_leg = np.zeros(len(sp), dtype=bool)
    for start, stop in splice.leg_windows:
        on_leg |= (seg_arc >= start) & (seg_arc <= stop)
    ax.plot(splice.points[:, 0], splice.points[:, 1], splice.points[:, 2],
            color="#b8b7ad", lw=2.0, label="QI curved return")
    legpts = splice.points[on_leg[:-1]]
    ax.scatter(legpts[:, 0], legpts[:, 1], legpts[:, 2], color=blue, s=6, label="straight mirror leg")
    ax.set_title("QI-mirror hybrid axis\n4 tangent-aligned legs + QI returns", fontsize=11, color=ink)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02))

    # (0,2) hybrid LCFS coloured by |B| (jet) ---------------------------------
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    axis_geom = setup.axis
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    radial = (np.cos(theta)[:, None, None] * np.asarray(axis_geom.normal)[None]
              + np.sin(theta)[:, None, None] * np.asarray(axis_geom.binormal)[None])
    boundary_b = np.asarray(mod_b[-1])
    dense_b = np.real(np.exp(1j * theta[:, None] * np.fft.fftfreq(boundary_b.shape[0], d=1.0 / boundary_b.shape[0])[None])
                      @ (np.fft.fft(boundary_b, axis=0) / boundary_b.shape[0]))
    surf_xyz = np.asarray(axis_geom.centerline)[None] + SECTION_RADIUS * radial
    norm_b = plt.Normalize(dense_b.min(), dense_b.max())
    ax.plot_surface(surf_xyz[..., 0], surf_xyz[..., 1], surf_xyz[..., 2],
                    facecolors=plt.cm.jet(norm_b(dense_b)), linewidth=0, antialiased=False, alpha=0.95)
    ax.set_title("Hybrid LCFS, $|B|$\n(circular section)", fontsize=11, color=ink)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_b, cmap="jet"), ax=ax, shrink=0.55, pad=0.11,
                 label="$|B|$")

    # (1,0) accuracy vs DOF ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    fdof = [r[1] for r in fourier_rows]; fmax = [r[3] for r in fourier_rows]; fleg = [r[4] for r in fourier_rows]
    bdof = [r[1] for r in bspline_rows]; bmax = [r[3] for r in bspline_rows]; bmid = [r[4] for r in bspline_rows]
    ax.loglog(fdof, fmax, "o-", color=orange, label="Fourier max error")
    ax.loglog(fdof, fleg, "o--", color=orange, alpha=0.55, label="Fourier ringing on leg")
    ax.loglog(bdof, bmax, "s-", color=blue, label="B-spline max error")
    ax.loglog(bdof, bmid, "s--", color=blue, alpha=0.7, label="B-spline leg midpoint")
    ax.set_xlabel("degrees of freedom"); ax.set_ylabel("axis error [m]")
    ax.set_title("Representation accuracy", fontsize=11, color=ink)
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    # (1,1) seam zoom: error vs arc length near a junction --------------------
    ax = fig.add_subplot(gs[1, 1])
    modes = fourier_rows[-1][0]; controls = bspline_rows[1][0]
    dense = dense_arc / splice.total_length * 2.0 * np.pi
    ffit = np.stack([fourier_fit(dense, target[:, j], modes) for j in range(3)], axis=1)
    bfit = bspline_fit(splice, controls, dense)
    ferr = np.linalg.norm(ffit - target, axis=1); berr = np.linalg.norm(bfit - target, axis=1)
    start, stop = splice.leg_windows[0]
    win = (dense_arc > start - 0.6) & (dense_arc < stop + 0.6)
    ax.axvspan(start, stop, color=grid_c, alpha=0.6, label="straight leg")
    ax.semilogy(dense_arc[win], ferr[win] + 1e-16, color=orange, lw=1.8, label=f"Fourier N={modes}")
    ax.semilogy(dense_arc[win], berr[win] + 1e-16, color=blue, lw=1.8, label=f"B-spline M={controls}")
    ax.set_xlabel("arc length [m]"); ax.set_ylabel("axis error [m]")
    ax.set_title("Seam: global ringing vs local exactness", fontsize=11, color=ink)
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper right")

    # (1,2) key numbers -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    rows = [
        ("QI reference (VMEC, Fourier)", ""),
        ("  |iota|  axis / edge", f"{abs(summary['qi_iota_axis']):.3f} / {abs(summary['qi_iota_edge']):.3f}"),
        ("  B0 / aspect", f"{summary['qi_b0']:.3f} / {summary['qi_aspect']:.2f}"),
        ("Cut curvature (min of axis)", f"{summary['cut_kappa'][0]:.2e} 1/m"),
        ("  curvature max / min", f"{summary['curvature_max'] / summary['curvature_min']:.0f}x"),
        ("Spliced axis closure", f"{summary['splice_closure']:.0e} m"),
        ("  legs (tangent break)", f"{len(summary['cut_phi'])} ({summary['corner_angle_deg']:.2f} deg)"),
        ("B-spline hybrid equilibrium", ""),
        ("  divergence rms", f"{summary['hybrid_divergence_rms']:.1e}"),
        ("  force normalized rms", f"{summary['hybrid_force_normalized_rms']:.2e}"),
        ("  rotational transform", f"{summary['hybrid_iota']:.3f}"),
        ("  |B| axis / LCFS max", f"{summary['hybrid_b_axis']:.2f} / {summary['hybrid_b_lcfs'][1]:.2f}"),
    ]
    y = 0.98
    for label, value in rows:
        weight = "bold" if value == "" else "normal"
        color = ink if value == "" else ink2
        ax.text(0.02, y, label, fontsize=9.5, color=color, fontweight=weight, va="top", transform=ax.transAxes)
        ax.text(0.98, y, value, fontsize=9.5, color=blue, va="top", ha="right",
                fontfamily="monospace", transform=ax.transAxes)
        y -= 0.083
    ax.set_title("QI vs QI-mirror hybrid", fontsize=11, color=ink, loc="left")

    fig.suptitle("QI-mirror hybrid: Fourier (global, Gibbs) vs B-spline (local, exact straight legs)",
                 fontsize=13.5, color=ink)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
