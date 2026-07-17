#!/usr/bin/env python3
"""Regenerate the README figures from the benchmark and golden-parity data.

Produces (into ``docs/_static/figures/``):

- ``readme_runtime_compare.png``      — VMEC2000 vs vmec_jax (cold/warm CPU,
  GPU where comparable) vs VMEC++, from ``benchmarks/baseline.json`` and
  ``benchmarks/gpu_baseline.json``.  Run ``benchmarks/run_baseline.py`` first.
- ``readme_convergence.png``          — force residual vs iteration for one
  representative case (nfp4_QH_warm_start at ns=51) in vmec_jax, VMEC2000
  (NSTEP=1 stdout trace), and VMEC++ (wout ``fsqt``).  Traces are cached in
  ``benchmarks/convergence_nfp4_ns51.json``; delete it to re-run the codes.
- ``readme_optimization.png``         — quasisymmetry (QA/QH/QP) seed vs
  optimized boundary cross-sections, 3-D LCFS geometry coloured by ``|B|``, and
  ``|B|`` in Boozer coordinates on the LCFS (jet line contours), from the decks
  in ``benchmarks/opt_decks/`` (R1 converged QA + regenerated QH/QP).  Each
  column is labelled with its measured QS residual.
- ``readme_qi.png``                   — quasi-isodynamic (QI) equilibria at
  nfp 1/2/3/4: boundary cross-sections, 3-D ``|B|`` geometry, and Boozer
  ``|B|`` (jet), from the bundled decks in ``examples/data/`` (``input.nfp1_QI``,
  ``input.nfp2_QI``, ``input.nfp3_QI_fixed_resolution_final``,
  ``input.nfp4_QI_finite_beta``).  Each column is labelled with its QI
  (omnigenity) residual.
- ``readme_precond.png``              — 2D block vs 1D radial preconditioner
  iteration counts on stiff cases (R10.2 measurements).
- ``readme_equilibrium_showcase.png`` — flux surfaces, 3-D boundary geometry
  coloured by ``|B|``, and ``|B|`` in Boozer coordinates on the LCFS (jet),
  for the bundled quick-start case (solves it in-process).
- ``readme_single_stage.png``         — cold-start single-stage vs two-stage
  plasma + coil optimization (vacuum and finite-beta columns): seed vs final
  boundaries with the headline ``B·n``/QS metrics of both approaches, and each
  approach's final LCFS coloured by ``|B|`` inside its own final coils.  Reads
  the outputs of ``examples/single_stage_vs_two_stage.py`` from
  ``output_single_stage_vs_two_stage/{vacuum,beta}/``; that experiment is
  multi-hour at full budget, so it is NEVER auto-run here — run it first.

Usage:
    python benchmarks/make_readme_figures.py
        [--only runtime,convergence,optimization,qi,precond,showcase,single_stage]
        [--outdir docs/_static/figures]

Figures are written uncompressed; compress before committing:
    magick <fig>.png -colors 255 png8:<fig>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"

# --------------------------------------------------------------------------
# Shared style (dataviz conventions: near-white surface, recessive chrome,
# colorblind-safe categorical slots, direct labels over chartjunk)
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"        # vmec_jax warm (the hero series)
BLUE_LIGHT = "#86b6ef"  # vmec_jax cold (same hue, lighter step)
YELLOW = "#eda100"      # VMEC++
VIOLET = "#4a3aa7"      # GPU
RED = "#e34948"
GREEN_TEXT = "#006300"


def apply_style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelsize": 10, "legend.frameon": False, "legend.fontsize": 9,
    })


CASE_LABELS = {
    "solovev": "Solovev",
    "DSHAPE": "DSHAPE",
    "circular_tokamak": "circular tokamak",
    "cth_like_fixed_bdy": "CTH-like",
    "li383_low_res": "LI383 (NCSX)",
    "LandremanPaul2021_QA_lowres": "Landreman-Paul QA",
    "LandremanPaul2021_QH_reactorScale_lowres": "Landreman-Paul QH reactor",
    "nfp4_QH_warm_start": "nfp4 QH",
    "NuhrenbergZille_1988_QHS": "Nuhrenberg-Zille QHS",
    "cth_like_free_bdy": "CTH-like free-bdy",
    "cth_like_free_bdy_lasym_small": "CTH-like free-bdy (lasym)",
}


# --------------------------------------------------------------------------
# 1. Runtime comparison
# --------------------------------------------------------------------------

def make_runtime_figure(out: Path) -> None:
    base = json.loads((REPO / "benchmarks" / "baseline.json").read_text())
    gpu = json.loads((REPO / "benchmarks" / "gpu_baseline.json").read_text())

    # GPU rows are single-grid first-stage runs; keep only rows whose
    # iteration count matches the CPU baseline (i.e. truly the same solve).
    gpu_rows = {
        case: vals["cuda/core_jit"]
        for case, vals in gpu.get("matrix", {}).items()
        if vals.get("cuda/core_jit", {}).get("ok")
    }

    rows = []
    for key, r in base.items():
        case, grid = key[:-1].split("[")
        v2k, cold = r.get("vmec2000", {}), r.get("vmec_jax_cold", {})
        warm, vpp = r.get("vmec_jax_warm", {}), r.get("vmecpp", {})
        if not (v2k.get("ok") and warm.get("ok")):
            continue
        label = CASE_LABELS.get(case, case)
        if grid == "multigrid":
            label += " (multigrid)"
        gr = None
        if grid == "input" and case in gpu_rows:
            if gpu_rows[case].get("iterations") == v2k.get("iterations"):
                gr = gpu_rows[case]["warm_wall_s"]
        rows.append(dict(label=label, ns=r.get("ns"), v2k=v2k.get("wall_s"),
                         cold=cold.get("wall_s"), warm=warm.get("warm_s"),
                         vpp=vpp.get("wall_s") if vpp.get("ok") else None,
                         gpu=gr))
    rows.sort(key=lambda r: r["v2k"])

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rows) + 1.7), dpi=160)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        xs = [v for v in (r["v2k"], r["cold"], r["warm"], r["vpp"], r["gpu"]) if v]
        ax.hlines(y, min(xs), max(xs), color=GRID, lw=1.0, zorder=1)

    mk = dict(s=52, zorder=3, linewidths=1.4, edgecolors=SURFACE)
    ax.scatter([r["v2k"] for r in rows], list(ys), color=INK2,
               label="VMEC2000 (Fortran)", **mk)
    ax.scatter([r["warm"] for r in rows], list(ys), color=BLUE,
               label="vmec_jax warm (in-process)", **mk)
    vpp_pts = [(r["vpp"], y) for y, r in zip(ys, rows) if r["vpp"]]
    ax.scatter([p[0] for p in vpp_pts], [p[1] for p in vpp_pts],
               color=YELLOW, label="VMEC++", **mk)
    gpu_pts = [(r["gpu"], y) for y, r in zip(ys, rows) if r["gpu"]]
    if gpu_pts:
        ax.scatter([p[0] for p in gpu_pts], [p[1] for p in gpu_pts],
                   color=VIOLET, marker="D", s=40, zorder=3, linewidths=1.4,
                   edgecolors=SURFACE, label="vmec_jax warm (GPU)")
    ax.scatter([r["cold"] for r in rows], list(ys), color=BLUE_LIGHT,
               label="vmec_jax cold (fresh CLI process)", **mk)

    xmax = max(max(v for v in (r["v2k"], r["cold"], r["vpp"] or 0, r["gpu"] or 0))
               for r in rows)
    for y, r in zip(ys, rows):
        sp = r["v2k"] / r["warm"]
        ax.annotate(f"{sp:,.0f}x" if sp >= 3 else f"{sp:.1f}x",
                    xy=(xmax * 2.6, y), va="center", ha="left",
                    fontsize=8.5, color=INK2, annotation_clip=False)
    ax.annotate("warm speedup\nvs VMEC2000", xy=(xmax * 2.45, len(rows) + 0.55),
                va="bottom", ha="left", fontsize=8, color=MUTED,
                annotation_clip=False)

    ax.set_xscale("log")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r['label']}  ·  ns={r['ns']}" for r in rows])
    ax.set_xlabel("wall-clock time (s, log scale)")
    ax.set_xlim(right=xmax * 2.4)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Full equilibrium solve: wall-clock time", loc="left",
                 pad=52, fontsize=13, color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncols=2,
              fontsize=8.5, columnspacing=1.4, handletextpad=0.25,
              borderaxespad=0.0, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 2. Parity table (solves the five golden-fixture cases)
# --------------------------------------------------------------------------

PARITY_CASES = {
    # case -> (multigrid?, README description)
    "solovev": (False, "2D analytic tokamak"),
    "DSHAPE": (True, "D-shaped tokamak, multigrid"),
    "circular_tokamak": (True, "high-ns tokamak, multigrid"),
    "li383_low_res": (False, "3D stellarator, nfp=3"),
    "nfp4_QH_warm_start": (False, "3D quasi-helical, nfp=4"),
}


def _golden_final_iteration(stdout_path: Path) -> int:
    final = None
    for line in stdout_path.read_text().splitlines():
        tok = line.split()
        if len(tok) >= 6 and tok[0].isdigit() and "E" in tok[1]:
            final = int(tok[0])
    if final is None:
        raise ValueError(f"no iteration rows in {stdout_path}")
    return final


def collect_parity() -> dict:
    import sys
    import netCDF4

    sys.path.insert(0, str(REPO / "tests"))
    from conftest import resolve_golden_dir

    from vmec_jax.core.input import VmecInput
    from vmec_jax.core import solver
    from vmec_jax.core.multigrid import solve_multigrid

    golden = resolve_golden_dir()
    if golden is None:
        raise RuntimeError("golden VMEC2000 fixtures unavailable")

    out = {}
    for name, (multigrid, _desc) in PARITY_CASES.items():
        inp = VmecInput.from_file(str(DATA / f"input.{name}"))
        res = solve_multigrid(inp) if multigrid else solver.solve(inp)
        with netCDF4.Dataset(golden / name / f"wout_{name}.nc") as ds:
            wb_gold = float(ds.variables["wb"][()])
        out[name] = dict(
            iterations=int(res.iterations),
            golden_iterations=_golden_final_iteration(golden / name / "stdout.txt"),
            jacobian_resets=int(res.jacobian_resets),
            wb_rel_err=abs(float(res.wb) - wb_gold) / abs(wb_gold),
        )
        print(name, out[name], flush=True)
    return out


def make_parity_figure(out: Path) -> None:
    data = collect_parity()
    rows = [(k, data[k]) for k in PARITY_CASES if k in data]

    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 1.9), dpi=160)
    ax.set_axis_off()
    cols = [0.02, 0.44, 0.60, 0.76, 0.99]
    ax.text(cols[0], 1.0, "case", fontsize=9.5, color=MUTED, va="center")
    ax.text(cols[1], 1.0, "VMEC2000\niterations", fontsize=9.5, color=MUTED,
            va="center", ha="center", linespacing=1.2)
    ax.text(cols[2], 1.0, "vmec_jax\niterations", fontsize=9.5, color=MUTED,
            va="center", ha="center", linespacing=1.2)
    ax.text(cols[3], 1.0, "match", fontsize=9.5, color=MUTED,
            va="center", ha="center")
    ax.text(cols[4], 1.0, "plasma energy wb\nrel. difference", fontsize=9.5,
            color=MUTED, va="center", ha="right", linespacing=1.2)

    n = len(rows)
    for i, (k, r) in enumerate(rows):
        y = 0.86 - 0.86 * (i + 0.5) / n
        if i % 2 == 0:
            ax.axhspan(y - 0.43 * 0.86 / n * 2, y + 0.43 * 0.86 / n * 2,
                       color="#f4f3ef", zorder=0)
        ax.text(cols[0], y, CASE_LABELS[k], fontsize=10.5, color=INK,
                va="center", fontweight="bold")
        ax.text(cols[0] + 0.205, y, PARITY_CASES[k][1], fontsize=8.5,
                color=MUTED, va="center")
        exact = r["iterations"] == r["golden_iterations"]
        ax.text(cols[1], y, f'{r["golden_iterations"]:,}', fontsize=11,
                color=INK2, va="center", ha="center", fontfamily="monospace")
        ax.text(cols[2], y, f'{r["iterations"]:,}', fontsize=11, color=BLUE,
                va="center", ha="center", fontfamily="monospace",
                fontweight="bold")
        ax.text(cols[3], y, "=" if exact else "-", fontsize=13,
                color=GREEN_TEXT if exact else RED, va="center", ha="center",
                fontweight="bold")
        wb_txt = "exact" if r["wb_rel_err"] == 0 else f'{r["wb_rel_err"]:.1e}'
        ax.text(cols[4], y, wb_txt, fontsize=11, color=INK2, va="center",
                ha="right", fontfamily="monospace")
        if r.get("jacobian_resets"):
            ax.text(cols[2] + 0.052, y, "*", fontsize=12, color=BLUE,
                    va="center", ha="center")

    ax.axhline(0.925, color=BASELINE, lw=0.8)
    ax.text(0.02, -0.10,
            "* including one mid-run jacobian reset, reproduced at the same iteration.",
            fontsize=8.5, color=MUTED, va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.08)
    ax.set_title("Iteration-for-iteration parity with VMEC2000", loc="left",
                 fontsize=13, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 3. Convergence trace: force residual vs iteration, three codes
# --------------------------------------------------------------------------

CONV_CASE = "nfp4_QH_warm_start"
CONV_NS = 51
CONV_CACHE = REPO / "benchmarks" / "convergence_nfp4_ns51.json"

VMECPP_TRACE_SNIPPET = r"""
import json, sys
import numpy as np
import vmecpp
out = vmecpp.run(vmecpp.VmecInput.from_file(sys.argv[1]), verbose=0)
# wout.fsqt is the per-iteration fsqr+fsqz+fsql trace (len == itfsq)
print(json.dumps({"niter": int(out.wout.niter),
                  "fsqt": np.asarray(out.wout.fsqt).tolist()}))
"""


def collect_convergence() -> dict:
    """Per-iteration total force residual (fsqr+fsqz+fsql) from all codes.

    - vmec_jax: ``SolveResult.fsq_history`` (recorded every iteration).
    - VMEC2000: stdout iteration table with NSTEP=1 (one row per iteration).
    - VMEC++: ``wout.fsqt`` (stored per iteration).
    Cached in CONV_CACHE; delete the file to re-run all three codes.
    """
    if CONV_CACHE.exists():
        return json.loads(CONV_CACHE.read_text())

    import re
    import subprocess
    import sys
    import tempfile

    sys.path.insert(0, str(REPO / "benchmarks"))
    from run_baseline import XVMEC2000, VMECPP_PY, make_ramped_deck

    from vmec_jax.core.input import VmecInput
    from vmec_jax.core import solver

    with tempfile.TemporaryDirectory() as td:
        deck = Path(td) / f"input.{CONV_CASE}"
        make_ramped_deck(DATA / f"input.{CONV_CASE}", deck, min_ns=CONV_NS)

        # vmec_jax: in-process solve, per-iteration history from SolveResult.
        res = solver.solve(VmecInput.from_file(str(deck)))
        hist = np.asarray(res.fsq_history)
        jax_fsq = hist[:, :3].sum(axis=1).tolist()

        # VMEC2000: NSTEP=1 makes the stdout table one row per iteration
        # (columns: ITER, FSQR, FSQZ, FSQL, RAX, DELT, WMHD).
        text = deck.read_text()
        deck.write_text(re.sub(r"NSTEP\s*=\s*\d+", "NSTEP = 1", text))
        proc = subprocess.run([str(XVMEC2000), deck.name], cwd=td,
                              capture_output=True, text=True, timeout=900)
        rows = re.findall(
            r"^\s*(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)",
            proc.stdout, re.M)
        v2k_fsq = [float(r[1]) + float(r[2]) + float(r[3]) for r in rows]

        # VMEC++: fsqt array from the wout payload.
        proc = subprocess.run([str(VMECPP_PY), "-c", VMECPP_TRACE_SNIPPET,
                               deck.name], cwd=td, capture_output=True,
                              text=True, timeout=900)
        vpp = json.loads(proc.stdout.strip().splitlines()[-1])

    data = {"case": CONV_CASE, "ns": CONV_NS,
            "ftol": 1e-13,  # deck FTOL_ARRAY final stage
            "vmec_jax": jax_fsq, "vmec2000": v2k_fsq, "vmecpp": vpp["fsqt"]}
    CONV_CACHE.write_text(json.dumps(data))
    return data


def make_convergence_figure(out: Path) -> None:
    d = collect_convergence()
    jax_t, v2k_t, vpp_t = d["vmec_jax"], d["vmec2000"], d["vmecpp"]

    fig, ax = plt.subplots(figsize=(8.6, 3.7), dpi=160)
    # Widest underneath, hero (blue) on top: the three traces coincide.
    ax.semilogy(range(1, len(v2k_t) + 1), v2k_t, color=INK2, lw=3.6,
                alpha=0.5, solid_capstyle="round",
                label=f"VMEC2000 (Fortran), {len(v2k_t)} iterations")
    ax.semilogy(range(1, len(vpp_t) + 1), vpp_t, color=YELLOW, lw=2.2,
                alpha=0.9, label=f"VMEC++, {len(vpp_t)} iterations")
    ax.semilogy(range(1, len(jax_t) + 1), jax_t, color=BLUE, lw=1.1,
                label=f"vmec_jax, {len(jax_t)} iterations")

    ax.axhline(3 * d["ftol"], color=BASELINE, lw=0.9, ls=(0, (5, 4)))
    ax.annotate("converged: fsqr, fsqz, fsql all < FTOL = 1e-13",
                xy=(len(jax_t) * 0.02, 3 * d["ftol"] * 1.6), ha="left",
                va="bottom", fontsize=8, color=MUTED)
    mid = len(jax_t) // 2
    ax.annotate("vmec_jax tracks VMEC2000\niteration-for-iteration\n"
                "(curves overlap)",
                xy=(mid, jax_t[mid]), xytext=(mid * 0.62, jax_t[mid] * 3e3),
                fontsize=8.5, color=INK2, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkB=4))

    ax.set_xlabel("iteration")
    ax.set_ylabel("force residual  fsqr + fsqz + fsql")
    ax.set_xlim(0, max(len(jax_t), len(v2k_t), len(vpp_t)) * 1.02)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Convergence trace: {CASE_LABELS[CONV_CASE]} "
                 f"(ns={d['ns']}, single grid)", loc="left", pad=10,
                 fontsize=13, color=INK)
    ax.legend(loc="upper right", fontsize=8.5, labelspacing=0.4,
              handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 4. Optimization panels.  Two figures, both with three rows (boundary, 3-D
#    LCFS coloured by |B|, and |B| in Boozer coordinates on the LCFS):
#      readme_optimization.png — quasisymmetry (QA/QH/QP), labelled with the
#        QS residual;
#      readme_qi.png           — quasi-isodynamic (QI) at nfp 1/2/3/4, labelled
#        with the QI (omnigenity) residual.
# --------------------------------------------------------------------------

# Optimized quasisymmetry decks live in benchmarks/opt_decks/ (the
# reproducibility inputs for readme_optimization.png): the genuine R1 converged
# QA deck fetched from the office campaign and the QH/QP decks regenerated by
# examples/optimization on a reduced continuation budget.  Each panel is
# labelled with the QS residual *measured here* on the shown equilibrium; the
# README table carries the full-campaign precise values.
OPT_DECKS = REPO / "benchmarks" / "opt_decks"

# quasisymmetry class -> (title, helicity (m, n), QS-surface count, family label)
OPT_CLASSES = [
    ("qa", "QA  ·  nfp 2", (1, 0), 10, "$|B| = |B|(s,\\theta)$"),
    ("qh", "QH  ·  nfp 4", (1, -1), 10, "$|B| = |B|(s,\\theta-N\\phi)$"),
    ("qp", "QP  ·  nfp 2", (0, 1), 10, "$|B| = |B|(s,\\phi)$"),
]

# quasi-isodynamic decks (bundled in examples/data), one column per field
# period, in nfp order.  These are the current best-available QI decks; the
# figure single-solves each and labels it with the QI (omnigenity) residual.
QI_CASES = [
    ("nfp1_QI", "QI  ·  nfp 1", 1),
    ("nfp2_QI", "QI  ·  nfp 2", 2),
    ("nfp3_QI_fixed_resolution_final", "QI  ·  nfp 3", 3),
    ("nfp4_QI_finite_beta", "QI  ·  nfp 4", 4),
]

QI_SURFACES = np.linspace(0.1, 1.0, 8)


def _input_boundary_rz(inp, theta: np.ndarray, phi: np.ndarray):
    """Boundary R, Z from a VmecInput's RBC/ZBS (no solve), shape (ntheta, nphi)."""
    rbc = np.asarray(inp.rbc, dtype=float)
    zbs = np.asarray(inp.zbs, dtype=float)
    ntor, nfp, mpol = int(inp.ntor), int(inp.nfp), rbc.shape[1]
    rbs = getattr(inp, "rbs", None)
    zbc = getattr(inp, "zbc", None)
    rbs = np.zeros_like(rbc) if rbs is None else np.asarray(rbs, dtype=float)
    zbc = np.zeros_like(zbs) if zbc is None else np.asarray(zbc, dtype=float)
    R = np.zeros((theta.size, phi.size))
    Z = np.zeros_like(R)
    for i in range(rbc.shape[0]):
        n = i - ntor
        for m in range(mpol):
            ang = m * theta[:, None] - n * nfp * phi[None, :]
            R += rbc[i, m] * np.cos(ang) + rbs[i, m] * np.sin(ang)
            Z += zbs[i, m] * np.sin(ang) + zbc[i, m] * np.cos(ang)
    return R, Z


def _plot_boundary_slices(axb, R, Z, *, color, seed=None):
    """Boundary cross-sections at the two phi slices in ``R``/``Z`` (ntheta, 2)."""
    if seed is not None:
        Rs, Zs = seed
        for k in range(Rs.shape[1]):
            axb.plot(Rs[:, k], Zs[:, k], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                     alpha=0.8, label="seed" if k == 0 else None)
    for k in range(R.shape[1]):
        if seed is not None:  # QS figure: seed vs a single "optimized" entry
            lab = "optimized" if k == 0 else None
        else:  # QI figure: distinguish the two phi slices
            lab = "$\\phi=0$" if k == 0 else ("half period" if k == 1 else None)
        axb.plot(R[:, k], Z[:, k], color=color, lw=1.7,
                 alpha=1.0 if k == 0 else 0.55, label=lab)
    # datalim (not box) keeps every panel the same box size — so titles and
    # labels align across a row of wildly different boundary scales — while
    # preserving true, undistorted equal-aspect cross-sections.
    axb.set_aspect("equal", adjustable="datalim")
    axb.tick_params(labelsize=7)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)


def _plot_3d_modB(fig, ax3d, wout, nfp):
    """3-D LCFS geometry coloured by |B| (same recipe as make_showcase_figure).

    Returns the (min, max) of |B| over the plotted surface.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    from vmec_jax.core.plotting import surface_modB, surface_rz

    ns = int(wout.ns)
    thg = np.linspace(0, 2 * np.pi, 64)
    phg = np.linspace(0, 2 * np.pi, min(240, 70 * nfp))
    Rg, Zg = surface_rz(wout, s_index=ns - 1, theta=thg, phi=phg)
    Bg = surface_modB(wout, s_index=ns - 1, theta=thg, phi=phg)
    phi2d = np.meshgrid(phg, thg)[0]
    Xg, Yg = Rg * np.cos(phi2d), Rg * np.sin(phi2d)
    Bn = (Bg - Bg.min()) / (Bg.max() - Bg.min() + 1e-30)
    ax3d.plot_surface(Xg, Yg, Zg, facecolors=cm.jet(Bn), rstride=1, cstride=1,
                      antialiased=False, linewidth=0.0, shade=False)
    scale = 0.55 * max(np.abs(Xg).max(), np.abs(Yg).max())
    try:
        ax3d.set_box_aspect((1, 1, 0.62), zoom=1.15)
    except TypeError:  # older matplotlib without the zoom kwarg
        ax3d.set_box_aspect((1, 1, 0.62))
    ax3d.auto_scale_xyz([-scale, scale], [-scale, scale],
                        [-scale * 0.62, scale * 0.62])
    ax3d.view_init(elev=30, azim=-55)
    ax3d.set_axis_off()
    sm = cm.ScalarMappable(cmap="jet",
                           norm=Normalize(float(Bg.min()), float(Bg.max())))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax3d, pad=0.0, fraction=0.045, shrink=0.62)
    cb.ax.tick_params(labelsize=6, colors=MUTED)
    cb.outline.set_visible(False)
    return float(Bg.min()), float(Bg.max())


def _plot_boozer_modB(fig, axm, wout, nfp, tag):
    """|B| in Boozer coordinates on the LCFS as jet line contours.

    Returns the (min, max) of |B| over the Boozer grid.
    """
    import tempfile

    import vmec_jax as vj
    from vmec_jax.core.boozer import run_booz_xform
    from vmec_jax.core.plotting import boozer_modB_on_surface

    with tempfile.TemporaryDirectory() as td:
        wp = vj.write_wout(Path(td) / f"wout_{tag}.nc", wout)
        bx = run_booz_xform(wp, mbooz=24, nbooz=24)
        tb, pb, B = boozer_modB_on_surface(bx, s_index=-1, ntheta=90, nphi=160)
    pc = axm.contour(pb * nfp / (2 * np.pi), tb / (2 * np.pi), B,
                     levels=22, cmap="jet", linewidths=0.8)
    cb = fig.colorbar(pc, ax=axm, pad=0.02, fraction=0.05)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    cb.outline.set_visible(False)
    axm.set_xlabel("$\\phi_B$ (field periods)", fontsize=8.5)
    axm.tick_params(labelsize=7)
    return float(B.min()), float(B.max())


def _compress_png(path: Path, max_kib: int = 400) -> None:
    """Palette-quantise to <=255 colours if oversized (like ``magick -colors 255``).

    Non-fatal: on any failure the (larger) figure is kept rather than the run
    aborted.  Dithering is disabled so text and thin contour lines stay crisp.
    """
    try:
        if path.stat().st_size <= max_kib * 1024:
            return
        from PIL import Image

        img = Image.open(path).convert("RGB").quantize(
            colors=255, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        img.save(path, optimize=True)
    except Exception as exc:  # pragma: no cover - best-effort compression
        print(f"  (compression skipped for {path.name}: {exc})", flush=True)


def make_optimization_figure(out: Path) -> None:
    import vmec_jax as vj
    from vmec_jax import optimize as opt

    present = [c for c in OPT_CLASSES if (OPT_DECKS / f"input.{c[0]}_optimized").exists()]
    if not present:
        raise FileNotFoundError(
            f"no optimized decks in {OPT_DECKS}; fetch/regenerate the R1 decks first")

    ncol = len(present)
    fig = plt.figure(figsize=(2.75 * ncol, 7.6), dpi=150)
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.0, 1.2, 1.05],
                          hspace=0.42, wspace=0.42)
    theta = np.linspace(0, 2 * np.pi, 241)

    for col, (tag, title, (hm, hn), nsurf, family) in enumerate(present):
        seed = vj.VmecInput.from_file(str(OPT_DECKS / f"input.{tag}_seed"))
        opt_inp = vj.VmecInput.from_file(str(OPT_DECKS / f"input.{tag}_optimized"))
        nfp = int(opt_inp.nfp)
        phi_arr = np.array([0.0, np.pi / nfp])  # phi = 0 and half field period

        # -- row 0: seed (grey) vs optimized (blue) boundary at two phi slices -
        axb = fig.add_subplot(gs[0, col])
        _plot_boundary_slices(
            axb, *_input_boundary_rz(opt_inp, theta, phi_arr), color=BLUE,
            seed=_input_boundary_rz(seed, theta, phi_arr))
        axb.set_title(title, loc="left", fontsize=10.5, color=INK, pad=3)
        if col == 0:
            axb.set_ylabel("Z (m)", fontsize=8.5)
            axb.legend(loc="upper right", fontsize=7, handlelength=1.5,
                       labelspacing=0.25, borderaxespad=0.1)

        # -- solve the optimized deck: measured QS + 3-D |B| + Boozer |B| ------
        try:
            eq = opt.solve_equilibrium(opt_inp)
        except Exception as exc:  # skip a column gracefully, keep the figure
            print(f"  {tag}: SOLVE FAILED ({exc}); skipping column", flush=True)
            axb.annotate("solve failed", xy=(0.5, -0.16), xycoords="axes fraction",
                         ha="center", va="top", fontsize=8.5, color=RED)
            continue
        qs_total = float(opt.QuasisymmetryRatioResidual(
            np.linspace(0.1, 1.0, nsurf), hm, hn).total(eq))

        ax3d = fig.add_subplot(gs[1, col], projection="3d")
        b_lo, b_hi = _plot_3d_modB(fig, ax3d, eq.wout, nfp)

        axm = fig.add_subplot(gs[2, col])
        _plot_boozer_modB(fig, axm, eq.wout, nfp, tag)
        if col == 0:
            axm.set_ylabel("$\\theta_B / 2\\pi$", fontsize=8.5)

        axb.annotate(f"QS = {qs_total:.2e}", xy=(0.5, -0.16),
                     xycoords="axes fraction", ha="center", va="top",
                     fontsize=8.5, color=GREEN_TEXT, fontweight="bold")
        axm.annotate(family, xy=(0.02, 1.02), xycoords="axes fraction",
                     ha="left", va="bottom", fontsize=7.5, color=MUTED)
        print(f"  {tag}: QS={qs_total:.3e} nfp={nfp} "
              f"|B|=[{b_lo:.2f},{b_hi:.2f}]T", flush=True)

    fig.suptitle("Quasisymmetry optimization from a circular seed: boundary, "
                 "3-D |B|, and Boozer |B| on the LCFS", x=0.5, ha="center",
                 fontsize=12.5, color=INK, y=0.995)
    fig.text(0.5, 0.945, "top: seed (grey, dashed) vs optimized (blue) boundary "
             "at $\\phi=0$ and a half field period   ·   middle: 3-D LCFS "
             "coloured by |B|   ·   bottom: |B| in Boozer coordinates (jet)",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _compress_png(out)
    print("wrote", out)


def make_qi_figure(out: Path) -> None:
    """Quasi-isodynamic (QI) equilibria at nfp 1/2/3/4: boundary, 3-D |B|, and
    Boozer |B| on the LCFS, each column labelled with the QI residual."""
    import vmec_jax as vj
    from vmec_jax import optimize as opt

    ncol = len(QI_CASES)
    fig = plt.figure(figsize=(2.6 * ncol, 7.6), dpi=150)
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.0, 1.2, 1.05],
                          hspace=0.42, wspace=0.42)
    theta = np.linspace(0, 2 * np.pi, 241)

    for col, (deck, title, _nfp_hint) in enumerate(QI_CASES):
        path = DATA / f"input.{deck}"
        if not path.exists():
            print(f"  {deck}: deck not found at {path}; skipping column", flush=True)
            continue
        inp = vj.VmecInput.from_file(str(path))
        nfp = int(inp.nfp)
        phi_arr = np.array([0.0, np.pi / nfp])  # phi = 0 and half field period

        # -- row 0: boundary at two phi slices (single equilibrium, no seed) ---
        axb = fig.add_subplot(gs[0, col])
        _plot_boundary_slices(axb, *_input_boundary_rz(inp, theta, phi_arr),
                              color=BLUE)
        axb.set_title(title, loc="left", fontsize=10.5, color=INK, pad=3)
        if col == 0:
            axb.set_ylabel("Z (m)", fontsize=8.5)
            axb.legend(loc="upper right", fontsize=7, handlelength=1.5,
                       labelspacing=0.25, borderaxespad=0.1)

        # -- solve the deck: QI (omnigenity) residual + 3-D |B| + Boozer |B| ---
        try:
            eq = opt.solve_equilibrium(inp)
        except Exception as exc:  # skip a column gracefully, keep the figure
            print(f"  {deck}: SOLVE FAILED ({exc}); skipping column", flush=True)
            axb.annotate("solve failed", xy=(0.5, -0.16), xycoords="axes fraction",
                         ha="center", va="top", fontsize=8.5, color=RED)
            continue
        qi_total = float(opt.quasi_isodynamic_residual_from_wout(
            eq.wout, surfaces=QI_SURFACES)["total"])

        ax3d = fig.add_subplot(gs[1, col], projection="3d")
        b_lo, b_hi = _plot_3d_modB(fig, ax3d, eq.wout, nfp)

        axm = fig.add_subplot(gs[2, col])
        _plot_boozer_modB(fig, axm, eq.wout, nfp, deck)
        if col == 0:
            axm.set_ylabel("$\\theta_B / 2\\pi$", fontsize=8.5)

        axb.annotate(f"QI = {qi_total:.2e}", xy=(0.5, -0.16),
                     xycoords="axes fraction", ha="center", va="top",
                     fontsize=8.5, color=GREEN_TEXT, fontweight="bold")
        print(f"  {deck}: QI={qi_total:.3e} nfp={nfp} "
              f"|B|=[{b_lo:.2f},{b_hi:.2f}]T", flush=True)

    fig.suptitle("Quasi-isodynamic (QI) equilibria across field periods "
                 "(nfp 1-4)", x=0.5, ha="center", fontsize=12.5, color=INK,
                 y=0.995)
    fig.text(0.5, 0.945, "top: boundary at $\\phi=0$ and a half field period "
             "  ·   middle: 3-D LCFS coloured by |B|   ·   bottom: |B| in "
             "Boozer coordinates (jet); label is the QI (omnigenity) residual",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _compress_png(out)
    print("wrote", out)


# --------------------------------------------------------------------------
# 5. 2D preconditioner: iteration reduction on stiff cases (R10.2)
# --------------------------------------------------------------------------

# Measured 2026-07-10 (plan.md R10.2, commit 2980d812): matrix-free 2D block
# preconditioner vs the default 1D radial preconditioner, iterations to the
# same FTOL on stiff decks.  Default 1D path stays byte-identical.
PRECOND_ROWS = [
    ("aspect-100 tokamak (a)", 97, 18),
    ("aspect-100 tokamak (b)", 163, 15),
    ("nfp4 QH, finite beta", 1885, 204),
]


def make_precond_figure(out: Path) -> None:
    rows = PRECOND_ROWS
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(rows) + 1.6), dpi=150)
    ys = np.arange(len(rows))[::-1]
    h = 0.34
    ax.barh(ys + h / 2, [r[1] for r in rows], height=h, color=BASELINE,
            label="1D radial preconditioner", zorder=2)
    ax.barh(ys - h / 2, [r[2] for r in rows], height=h, color=BLUE,
            label="2D block preconditioner", zorder=2)
    for y, (_lab, one_d, two_d) in zip(ys, rows):
        ax.annotate(f"{one_d}", xy=(one_d, y + h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=INK2)
        ax.annotate(f"{two_d}   ({one_d / two_d:.1f}x fewer)",
                    xy=(two_d, y - h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=BLUE, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("iterations to converge (lower is better)")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.28)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("2D block preconditioner: iteration reduction on stiff cases",
                 loc="left", pad=30, fontsize=12.5, color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=2,
              fontsize=8.5, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 6. Equilibrium showcase (solves the bundled quick-start case)
# --------------------------------------------------------------------------

def make_showcase_figure(out: Path) -> None:
    import tempfile

    import vmec_jax as vj
    from vmec_jax.core.input import VmecInput
    from vmec_jax.core.multigrid import solve_multigrid
    from vmec_jax.core.wout import wout_from_state
    from vmec_jax.core.boozer import run_booz_xform
    from vmec_jax.core.plotting import (surface_rz, surface_modB, axis_rz,
                                        boozer_modB_on_surface)
    from matplotlib import cm
    from matplotlib.colors import Normalize

    inp = VmecInput.from_file(str(DATA / "input.nfp4_QH_warm_start"))
    res = solve_multigrid(inp)
    wout = wout_from_state(inp=inp, state=res.state, niter=res.iterations,
                           fsqr=res.fsqr, fsqz=res.fsqz, fsql=res.fsql)
    ns, nfp = int(wout.ns), int(wout.nfp)

    fig = plt.figure(figsize=(9.6, 3.5), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.35], wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax2 = fig.add_subplot(gs[0, 2])

    # -- panel 1: flux-surface cross sections ------------------------------
    theta = np.linspace(0, 2 * np.pi, 181)
    phis = [0.0, np.pi / (2 * nfp), np.pi / nfp]
    phi_labels = ["$\\phi = 0$", "quarter period", "half period"]
    ramp = ["#9ec5f4", "#5598e7", "#1c5cab"]
    for phi, c, lab in zip(phis, ramp, phi_labels):
        for j, f in enumerate((0.25, 0.5, 1.0)):
            si = max(1, round((ns - 1) * f))
            R, Z = surface_rz(wout, s_index=si, theta=theta, phi=np.array([phi]))
            ax1.plot(R[:, 0], Z[:, 0], color=c, lw=1.7 if f == 1.0 else 0.85,
                     alpha=1.0 if f == 1.0 else 0.6,
                     label=lab if j == 2 else None)
        Ra, Za = axis_rz(wout, np.array([phi]))
        ax1.plot(Ra, Za, marker="o", ms=3.5, color=c, mec=SURFACE, mew=1.0,
                 ls="none")
    ax1.set_aspect("equal")
    ax1.set_xlabel("R (m)", fontsize=9)
    ax1.set_ylabel("Z (m)", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=7,
               handlelength=1.2, labelspacing=0.3)
    ax1.set_title("Flux surfaces", loc="left", fontsize=10.5, color=INK)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # -- panel 2: 3-D boundary geometry coloured by |B| --------------------
    thg = np.linspace(0, 2 * np.pi, 80)
    phg = np.linspace(0, 2 * np.pi, min(360, 90 * nfp))
    Rg, Zg = surface_rz(wout, s_index=ns - 1, theta=thg, phi=phg)
    Bg = surface_modB(wout, s_index=ns - 1, theta=thg, phi=phg)
    phi2d = np.meshgrid(phg, thg)[0]
    Xg, Yg = Rg * np.cos(phi2d), Rg * np.sin(phi2d)
    Bn = (Bg - Bg.min()) / (Bg.max() - Bg.min() + 1e-30)
    ax3d.plot_surface(Xg, Yg, Zg, facecolors=cm.jet(Bn), rstride=1, cstride=1,
                      antialiased=False, linewidth=0.0, shade=False)
    scale = 0.55 * max(np.abs(Xg).max(), np.abs(Yg).max())
    try:
        ax3d.set_box_aspect((1, 1, 0.62), zoom=1.18)
    except TypeError:  # older matplotlib without the zoom kwarg
        ax3d.set_box_aspect((1, 1, 0.62))
    ax3d.auto_scale_xyz([-scale, scale], [-scale, scale],
                        [-scale * 0.62, scale * 0.62])
    ax3d.view_init(elev=30, azim=-55)
    ax3d.set_axis_off()
    sm = cm.ScalarMappable(cmap="jet",
                           norm=Normalize(float(Bg.min()), float(Bg.max())))
    sm.set_array([])
    cb0 = fig.colorbar(sm, ax=ax3d, pad=0.02, fraction=0.04, shrink=0.58)
    cb0.ax.tick_params(labelsize=7, colors=MUTED)
    cb0.outline.set_visible(False)
    ax3d.text2D(0.5, 1.04, "3-D boundary,  |B| (T)", transform=ax3d.transAxes,
                ha="center", va="bottom", fontsize=10.5, color=INK)

    # -- panel 3: |B| in Boozer coordinates on the LCFS (jet) --------------
    with tempfile.TemporaryDirectory() as td:
        wp = vj.write_wout(Path(td) / "wout_showcase.nc", wout)
        bx = run_booz_xform(wp, mbooz=28, nbooz=28)
        tb, pb, Bb = boozer_modB_on_surface(bx, s_index=-1, ntheta=161, nphi=161)
    pc = ax2.contour(pb * nfp / (2 * np.pi), tb / (2 * np.pi), Bb,
                     levels=24, cmap="jet", linewidths=0.9)
    cb = fig.colorbar(pc, ax=ax2, pad=0.02)
    cb.set_label("|B| (T)", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)
    ax2.set_xlabel("Boozer toroidal angle (field periods)", fontsize=9)
    ax2.set_ylabel("Boozer poloidal angle / 2$\\pi$", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.set_title("|B| in Boozer coordinates (LCFS)", loc="left", fontsize=10.5,
                  color=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 7. Cold-start single-stage vs two-stage plasma + coil optimization.  Reads
#    the outputs of examples/single_stage_vs_two_stage.py from
#    output_single_stage_vs_two_stage/{vacuum,beta}/: wout_stage1.nc /
#    wout_single.nc (the two final boundaries), coils_stage2.npz /
#    coils_single.npz (final coil curve dofs + currents, reconstructed via
#    ESSOS), and comparison.json (the honest evaluate-phase metrics).  The
#    experiment is multi-hour at full budget, so it is NEVER auto-run here.
# --------------------------------------------------------------------------

SINGLE_STAGE_OUT = REPO / "output_single_stage_vs_two_stage"
SINGLE_STAGE_SEED = DATA / "input.minimal_seed_nfp2"
SINGLE_STAGE_CASES = [
    ("vacuum", "QA  ·  vacuum"),
    ("beta", "QA  ·  finite $\\beta$"),
]


def _coil_gamma_from_npz(npz_path: Path) -> np.ndarray:
    """Full coil-filament set (n_coils, n_segments, 3) from saved curve dofs.

    ``examples/single_stage_vs_two_stage.py`` saves only the base-coil curve
    dofs + currents (keys ``cdofs``, ``currents``, ``nfp``, ``n_segments``);
    ESSOS rebuilds the nfp / stellarator-symmetry copies.
    """
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    d = np.load(npz_path)
    curves = Curves(jnp.asarray(d["cdofs"]), int(d["n_segments"]),
                    int(d["nfp"]), True)
    return np.asarray(Coils(curves, jnp.asarray(d["currents"])).gamma)


def _plot_coils_and_lcfs(fig, ax3d, wout, gamma, nfp):
    """Final LCFS coloured by |B| inside the fixed ESSOS coil filaments.

    Same |B| surface recipe as :func:`_plot_3d_modB`, but the view is scaled to
    the (larger) coil bounding box and the ESSOS filaments (``coils.gamma``, one
    closed loop per coil) are overlaid.  Returns the (min, max) of |B|.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    from vmec_jax.core.plotting import surface_modB, surface_rz

    ns = int(wout.ns)
    thg = np.linspace(0, 2 * np.pi, 64)
    phg = np.linspace(0, 2 * np.pi, min(240, 70 * nfp))
    Rg, Zg = surface_rz(wout, s_index=ns - 1, theta=thg, phi=phg)
    Bg = surface_modB(wout, s_index=ns - 1, theta=thg, phi=phg)
    phi2d = np.meshgrid(phg, thg)[0]
    Xg, Yg = Rg * np.cos(phi2d), Rg * np.sin(phi2d)
    Bn = (Bg - Bg.min()) / (Bg.max() - Bg.min() + 1e-30)
    ax3d.plot_surface(Xg, Yg, Zg, facecolors=cm.jet(Bn), rstride=1, cstride=1,
                      antialiased=False, linewidth=0.0, shade=False)
    for k in range(gamma.shape[0]):  # ESSOS filaments, closed loops (copper)
        g = np.vstack([gamma[k], gamma[k, :1]])
        ax3d.plot(g[:, 0], g[:, 1], g[:, 2], color="#b06a34", lw=1.0, alpha=0.9)
    scale = 1.03 * float(max(np.abs(gamma[:, :, 0]).max(), np.abs(gamma[:, :, 1]).max()))
    zmax = 1.05 * float(np.abs(gamma[:, :, 2]).max())
    try:
        ax3d.set_box_aspect((1, 1, zmax / scale), zoom=1.5)
    except TypeError:  # older matplotlib without the zoom kwarg
        ax3d.set_box_aspect((1, 1, zmax / scale))
    ax3d.auto_scale_xyz([-scale, scale], [-scale, scale], [-zmax, zmax])
    ax3d.view_init(elev=32, azim=-60)
    ax3d.set_axis_off()
    sm = cm.ScalarMappable(cmap="jet",
                           norm=Normalize(float(Bg.min()), float(Bg.max())))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax3d, pad=0.0, fraction=0.04, shrink=0.6)
    cb.ax.tick_params(labelsize=6, colors=MUTED)
    cb.outline.set_visible(False)
    return float(Bg.min()), float(Bg.max())


def make_single_stage_figure(out: Path) -> None:
    import vmec_jax as vj
    from vmec_jax.core.plotting import surface_rz

    cases = []
    for name, title in SINGLE_STAGE_CASES:
        cdir = SINGLE_STAGE_OUT / name
        need = [cdir / "comparison.json", cdir / "wout_stage1.nc",
                cdir / "wout_single.nc", cdir / "coils_stage2.npz",
                cdir / "coils_single.npz"]
        missing = [p for p in need if not p.exists()]
        if missing:
            print(f"  {name}: missing {missing[0].relative_to(REPO)} -- run "
                  f"'python examples/single_stage_vs_two_stage.py --case {name} "
                  "--phase all' first (multi-hour at full budget); skipping "
                  "this column", flush=True)
            continue
        cases.append((name, title, cdir))
    if not cases:
        print("  no single-stage outputs -- figure skipped")
        return

    seed_inp = vj.VmecInput.from_file(str(SINGLE_STAGE_SEED))
    ncol = len(cases)
    fig = plt.figure(figsize=(3.4 * ncol, 9.4), dpi=150)
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.0, 0.92, 0.92],
                          hspace=0.24, wspace=0.28)
    theta = np.linspace(0, 2 * np.pi, 241)

    for col, (name, title, cdir) in enumerate(cases):
        comp = json.loads((cdir / "comparison.json").read_text())
        ts, ss = comp["two_stage"], comp["single_stage"]
        w_two = vj.read_wout(cdir / "wout_stage1.nc")     # two-stage boundary
        w_sin = vj.read_wout(cdir / "wout_single.nc")     # single-stage boundary
        nfp = int(w_sin.nfp)
        phi_arr = np.array([0.0, np.pi / nfp])  # phi = 0 and half field period

        # -- row 0: seed vs two-stage vs single-stage boundary -----------------
        # All three on one axes: the shared (auto-scaled, equal-aspect) limits
        # keep the crude cold-start seed AND both finals visible together.
        axb = fig.add_subplot(gs[0, col])
        Rs, Zs = _input_boundary_rz(seed_inp, theta, phi_arr)
        Rt, Zt = surface_rz(w_two, s_index=-1, theta=theta, phi=phi_arr)
        Rf, Zf = surface_rz(w_sin, s_index=-1, theta=theta, phi=phi_arr)
        for k in range(phi_arr.size):
            axb.plot(Rs[:, k], Zs[:, k], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                     alpha=0.85, label="seed" if k == 0 else None)
        for k in range(phi_arr.size):
            axb.plot(Rt[:, k], Zt[:, k], color=YELLOW, lw=1.6,
                     alpha=1.0 if k == 0 else 0.55,
                     label="two-stage" if k == 0 else None)
        for k in range(phi_arr.size):
            axb.plot(Rf[:, k], Zf[:, k], color=BLUE, lw=1.6,
                     alpha=1.0 if k == 0 else 0.55,
                     label="single-stage" if k == 0 else None)
        axb.set_aspect("equal", adjustable="datalim")
        axb.set_title(title, loc="left", fontsize=10.5, color=INK, pad=3)
        axb.tick_params(labelsize=7)
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        if col == 0:
            axb.set_ylabel("Z (m)", fontsize=8.5)
        axb.set_xlabel("R (m)", fontsize=8.5)
        axb.legend(loc="upper right", fontsize=6.5, handlelength=1.4,
                   labelspacing=0.25, borderaxespad=0.1)
        pol = comp.get("single_stage_polish")
        bn_line = ("$\\langle|B\\cdot n|\\rangle/\\langle B\\rangle$: "
                   f"{ts['avg_Bn_over_B']:.1e}"
                   + (f" $\\to$ +polish {pol['avg_Bn_over_B']:.1e}" if pol else "")
                   + f" (cold {ss['avg_Bn_over_B']:.1e})")
        qs_line = (f"QS: {ts['qs_total']:.1e}"
                   + (f" $\\to$ {pol['qs_total']:.1e}" if pol else "")
                   + f" (cold {ss['qs_total']:.1e})")
        axb.annotate(
            bn_line + "\n" + qs_line,
            xy=(0.5, -0.24), xycoords="axes fraction", ha="center", va="top",
            fontsize=7, color=GREEN_TEXT, fontweight="bold")
        if name == "beta":
            # The case's calibrated seed <beta> (the finals' betatotal shifts
            # with each approach's plasma volume at the fixed pres_scale).
            seed_meta = cdir / "seed.json"
            if seed_meta.exists():
                beta_pct = float(json.loads(seed_meta.read_text())["beta_pct"])
            else:
                beta_pct = 100.0 * float(np.mean([float(w_two.betatotal),
                                                  float(w_sin.betatotal)]))
            axb.annotate(f"$\\langle\\beta\\rangle \\approx$ {beta_pct:.1f}%",
                         xy=(0.98, 1.02), xycoords="axes fraction", ha="right",
                         va="bottom", fontsize=8, color=MUTED)

        # -- rows 1-2: each approach's final LCFS |B| inside its final coils ---
        for row, (wout, npz, lab, c) in enumerate(
                [(w_two, "coils_stage2.npz", "two-stage", YELLOW),
                 (w_sin, "coils_single.npz", "single-stage", BLUE)], start=1):
            gamma = _coil_gamma_from_npz(cdir / npz)
            ax3d = fig.add_subplot(gs[row, col], projection="3d")
            ax3d.patch.set_alpha(0.0)  # keep row-0's metric annotation visible
            b_lo, b_hi = _plot_coils_and_lcfs(fig, ax3d, wout, gamma, nfp)
            ax3d.text2D(0.03, 0.10, lab, transform=ax3d.transAxes, ha="left",
                        va="bottom", fontsize=9, color=c, fontweight="bold")
            print(f"  {name}/{lab}: <|B.n|>/<B>={comp[lab.replace('-', '_')]['avg_Bn_over_B']:.3e}  "
                  f"QS={comp[lab.replace('-', '_')]['qs_total']:.3e}  "
                  f"|B|=[{b_lo:.2f},{b_hi:.2f}]T", flush=True)

    fig.suptitle("Cold-start single-stage vs two-stage plasma + coil "
                 "optimization", x=0.5, ha="center", fontsize=12.5, color=INK,
                 y=0.995)
    fig.text(0.5, 0.966, "top: seed (grey, dashed) vs two-stage (orange) vs "
             "single-stage (blue) boundary at $\\phi=0$ and a half field period",
             ha="center", fontsize=8, color=MUTED)
    fig.text(0.5, 0.951, "middle / bottom: each approach's final LCFS "
             "coloured by |B| inside its own final coils",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.936))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _compress_png(out)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default="runtime,convergence,optimization,qi,precond,showcase,single_stage")
    ap.add_argument("--outdir", default=str(REPO / "docs" / "_static" / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style()
    which = set(args.only.split(","))
    if "runtime" in which:
        make_runtime_figure(outdir / "readme_runtime_compare.png")
    if "convergence" in which:
        make_convergence_figure(outdir / "readme_convergence.png")
    if "optimization" in which:
        make_optimization_figure(outdir / "readme_optimization.png")
    if "qi" in which:
        make_qi_figure(outdir / "readme_qi.png")
    if "precond" in which:
        make_precond_figure(outdir / "readme_precond.png")
    if "showcase" in which:
        make_showcase_figure(outdir / "readme_equilibrium_showcase.png")
    if "single_stage" in which:
        make_single_stage_figure(outdir / "readme_single_stage.png")


if __name__ == "__main__":
    main()
