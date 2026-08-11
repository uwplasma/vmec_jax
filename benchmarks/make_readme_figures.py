#!/usr/bin/env python3
"""Regenerate the small set of benchmark figures used by the documentation.

Produces (into ``docs/_static/figures/``):

- ``readme_runtime_compare.png``      — VMEC2000 vs vmex (cold/warm CPU,
  GPU where comparable) vs the reference C++ implementation, from
  ``benchmarks/baseline.json`` and
  ``benchmarks/gpu_baseline.json``.  Run ``benchmarks/run_baseline.py`` first.
- ``readme_convergence.png``          — force residual vs iteration for one
  representative case (nfp4_QH_warm_start at ns=51) in vmex, VMEC2000
  (NSTEP=1 stdout trace), and the reference C++ implementation (wout
  ``fsqt``).  Traces are cached in
  ``benchmarks/convergence_nfp4_ns51.json``; delete it to re-run the codes.
- ``readme_precond.png``              — 2D block vs 1D radial preconditioner
  iteration counts on stiff cases (R10.2 measurements).
- ``readme_equilibrium_showcase.png`` — flux surfaces, 3-D boundary geometry
  coloured by ``|B|``, and ``|B|`` in Boozer coordinates on the LCFS (jet),
  for the bundled quick-start case (solves it in-process).
Usage:
    python benchmarks/make_readme_figures.py
        [--only runtime,convergence,precond,showcase]
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
BLUE = "#2a78d6"        # vmex warm (the hero series)
BLUE_LIGHT = "#86b6ef"  # vmex cold (same hue, lighter step)
YELLOW = "#eda100"      # reference C++
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
        if key.startswith("_"):
            continue
        case, grid = key[:-1].split("[")
        v2k, cold = r.get("vmec2000", {}), r.get("vmex_cold", {})
        warm, vpp = r.get("vmex_warm", {}), r.get("vmecpp", {})
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

    ns_set = {r["ns"] for r in rows}
    same_ns = len(ns_set) == 1

    fig, ax = plt.subplots(figsize=(9.6, 0.50 * len(rows) + 2.1), dpi=160)
    fig.subplots_adjust(left=0.255, right=0.865, top=0.845, bottom=0.105)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f3ef", zorder=0)
        xs = [v for v in (r["v2k"], r["cold"], r["warm"], r["vpp"], r["gpu"]) if v]
        ax.hlines(y, min(xs), max(xs), color=BASELINE, lw=2.2, zorder=1,
                  capstyle="round")

    # hero (warm, blue) drawn last so it stays on top where points coincide
    mk = dict(s=120, zorder=3, linewidths=1.8, edgecolors="white")
    ax.scatter([r["cold"] for r in rows], list(ys), color=BLUE_LIGHT,
               label="vmex cold (fresh CLI process)", **mk)
    vpp_pts = [(r["vpp"], y) for y, r in zip(ys, rows) if r["vpp"]]
    ax.scatter([p[0] for p in vpp_pts], [p[1] for p in vpp_pts],
               color=YELLOW, label="reference C++", **mk)
    gpu_pts = [(r["gpu"], y) for y, r in zip(ys, rows) if r["gpu"]]
    if gpu_pts:
        ax.scatter([p[0] for p in gpu_pts], [p[1] for p in gpu_pts],
                   color=VIOLET, marker="D", s=95, zorder=3, linewidths=1.8,
                   edgecolors="white", label="vmex warm (GPU)")
    ax.scatter([r["v2k"] for r in rows], list(ys), color=INK2,
               label="VMEC2000 (Fortran)", **mk)
    ax.scatter([r["warm"] for r in rows], list(ys), color=BLUE, zorder=4,
               s=120, linewidths=1.8, edgecolors="white",
               label="vmex warm (in-process)")

    xmax = max(max(v for v in (r["v2k"], r["cold"], r["vpp"] or 0, r["gpu"] or 0))
               for r in rows)
    tr = ax.get_yaxis_transform()  # x in axes fraction, y in data
    for y, r in zip(ys, rows):
        sp = r["v2k"] / r["warm"]
        ax.text(1.045, y, f"{sp:,.0f}×" if sp >= 3 else f"{sp:.1f}×",
                transform=tr, va="center", ha="center", fontsize=11,
                color=BLUE, fontweight="bold", clip_on=False)
    ax.text(1.045, 1.015, "warm speedup\nvs VMEC2000", transform=ax.transAxes,
            va="bottom", ha="center", fontsize=9, color=MUTED,
            linespacing=1.35)

    ax.set_xscale("log")
    ax.set_yticks(list(ys))
    if same_ns:
        ax.set_yticklabels([r["label"] for r in rows], fontsize=10.5)
    else:
        ax.set_yticklabels([f"{r['label']}  ·  ns={r['ns']}" for r in rows],
                           fontsize=10.5)
    ax.set_xlabel("wall-clock time (s, log scale)  —  left is faster",
                  fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlim(right=xmax * 1.7)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    title = "Full equilibrium solve: wall-clock time"
    if same_ns:
        title += f"  (all cases ns={next(iter(ns_set))})"
    ax.set_title(title, loc="left", pad=54, fontsize=14, color=INK)
    handles, labels = ax.get_legend_handles_labels()
    order = ["VMEC2000 (Fortran)", "reference C++",
             "vmex warm (in-process)", "vmex cold (fresh CLI process)",
             "vmex warm (GPU)"]
    pairs = sorted(zip(handles, labels), key=lambda hl: order.index(hl[1]))
    ax.legend([p[0] for p in pairs], [p[1] for p in pairs],
              loc="lower left", bbox_to_anchor=(-0.02, 1.005), ncols=2,
              fontsize=10, columnspacing=1.6, handletextpad=0.15,
              borderaxespad=0.0, labelspacing=0.45)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 2. Convergence trace: force residual vs iteration, three codes
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

    - vmex: ``SolveResult.fsq_history`` (recorded every iteration).
    - VMEC2000: stdout iteration table with NSTEP=1 (one row per iteration).
    - reference C++: ``wout.fsqt`` (stored per iteration).
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

    from vmex.core.input import VmecInput
    from vmex.core import solver

    with tempfile.TemporaryDirectory() as td:
        deck = Path(td) / f"input.{CONV_CASE}"
        make_ramped_deck(DATA / f"input.{CONV_CASE}", deck, min_ns=CONV_NS)

        # vmex: in-process solve, per-iteration history from SolveResult.
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

        # reference C++: fsqt array from the wout payload.
        proc = subprocess.run([str(VMECPP_PY), "-c", VMECPP_TRACE_SNIPPET,
                               deck.name], cwd=td, capture_output=True,
                              text=True, timeout=900)
        vpp = json.loads(proc.stdout.strip().splitlines()[-1])

    data = {"case": CONV_CASE, "ns": CONV_NS,
            "ftol": 1e-13,  # deck FTOL_ARRAY final stage
            "vmex": jax_fsq, "vmec2000": v2k_fsq, "vmecpp": vpp["fsqt"]}
    CONV_CACHE.write_text(json.dumps(data))
    return data


def make_convergence_figure(out: Path) -> None:
    d = collect_convergence()
    jax_t, v2k_t, vpp_t = d["vmex"], d["vmec2000"], d["vmecpp"]

    fig, ax = plt.subplots(figsize=(8.6, 3.7), dpi=160)
    # Widest underneath, hero (blue) on top: the three traces coincide.
    ax.semilogy(range(1, len(v2k_t) + 1), v2k_t, color=INK2, lw=3.6,
                alpha=0.5, solid_capstyle="round",
                label=f"VMEC2000 (Fortran), {len(v2k_t)} iterations")
    ax.semilogy(range(1, len(vpp_t) + 1), vpp_t, color=YELLOW, lw=2.2,
                alpha=0.9, label=f"reference C++, {len(vpp_t)} iterations")
    ax.semilogy(range(1, len(jax_t) + 1), jax_t, color=BLUE, lw=1.1,
                label=f"vmex, {len(jax_t)} iterations")

    ax.axhline(3 * d["ftol"], color=BASELINE, lw=0.9, ls=(0, (5, 4)))
    ax.annotate("converged: fsqr, fsqz, fsql all < FTOL = 1e-13",
                xy=(len(jax_t) * 0.02, 3 * d["ftol"] * 1.6), ha="left",
                va="bottom", fontsize=8, color=MUTED)
    mid = len(jax_t) // 2
    ax.annotate("vmex tracks VMEC2000\niteration-for-iteration\n"
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
# 3. 2D preconditioner: iteration reduction on stiff cases (R10.2)
# --------------------------------------------------------------------------

# Measured 2026-07-10 (R10.2, commit 2980d812): matrix-free 2D block
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

    import vmex as vj
    from vmex.core.input import VmecInput
    from vmex.core.multigrid import solve_multigrid
    from vmex.core.wout import wout_from_state
    from vmex.core.boozer import run_booz_xform
    from vmex.core.plotting import (surface_rz, surface_modB, axis_rz,
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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="runtime,convergence,precond,showcase")
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
    if "precond" in which:
        make_precond_figure(outdir / "readme_precond.png")
    if "showcase" in which:
        make_showcase_figure(outdir / "readme_equilibrium_showcase.png")


if __name__ == "__main__":
    main()
