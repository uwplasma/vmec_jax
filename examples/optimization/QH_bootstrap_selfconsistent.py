#!/usr/bin/env python
"""Bootstrap-self-consistent QH (nfp=4, aspect 6.5, beta=2.5%) of arXiv:2205.02914.

Reproduces the quasi-helically-symmetric configuration with self-consistent
bootstrap current of Landreman, Buller & Drevlak, Phys. Plasmas 29, 082501
(2022), arXiv:2205.02914, from the paper's Zenodo data archive: load the
published boundary + pressure deck, *erase* the current profile (CURTOR = 0,
flat I'), and let the fixed-boundary Picard loop ``self_consistent_bootstrap``
regenerate it from the Redl formula [Redl et al., Phys. Plasmas 28, 022502
(2021)] with the paper's kinetic profiles ne = 2.2e20*(1 - s^5) m^-3, Te = Ti =
10 keV*(1 - s), Zeff = 1 (the deck's pressure is exactly e*(ne*Te + ni*Ti)).
The QH Redl isomorphism uses ``helicity_n = -1`` (``iota -> iota - nfp*n``).
Achieved 2026-07-12 (full mode, deck mpol=16/ntor=12, ns 13->25->51): 4 Picard
iterations to f_boot = 7.5e-06, I_p = -1.211 MA vs the published
CURTOR = -1.208 MA (0.3%), <J.B> vs the published profile 0.93% RMS.  (The
paper's QH SFINCS points are a PDF-only figure in the archive, so the SFINCS
overlay is shown for QA only.)  Needs the Zenodo dataset on disk (default path
as in tests/test_bootstrap.py; override with VMEX_ZENODO_2205_02914); the
sibling ``QA_bootstrap_selfconsistent.py`` and this script assemble the
combined two-panel ``readme_bootstrap.png`` once both have run.
"""

import dataclasses
import os
from pathlib import Path

import numpy as np

import vmex as vj
from vmex.core import bootstrap as bs

# --------------------------- parameters ------------------------------------
ZENODO = Path(os.environ.get(
    "VMEX_ZENODO_2205_02914",
    "/Users/rogerio/local/"
    "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current"))
CONFIG_DIR = ZENODO / "configurations" / "QH_aspect6.5_beta2.5"
_STEM = "20220218-01-021_QH_A6.5_n0_2.2_T0_10_highResVmecForBestFrom020"
DECK = CONFIG_DIR / f"input.{_STEM}"
WOUT_PUB = CONFIG_DIR / f"wout_{_STEM}.nc"
TAG, TITLE = "QH", "QH   nfp=4, aspect 6.5, beta=2.5%"
N0, T0 = 2.2e20, 10.0e3         # ne = N0*(1-s^5) [1/m^3], Te = Ti = T0*(1-s) [eV]
HELICITY_N = -1                 # quasi-helical: Redl isomorphism iota -> iota - nfp*helicity_n
NS_ARRAY = [13, 25, 51]         # radial ladder (published deck ran to ns=201)
MAX_MODE = None                 # boundary truncation; None = deck resolution
N_ITER, TOL = 10, 1e-3          # Picard budget / I'(s) convergence tolerance
OUT_DIR = Path(f"output_{TAG}_bootstrap_selfconsistent")
if os.environ.get("VMEX_EXAMPLES_CI") == "1":   # smoke-test budget
    NS_ARRAY, MAX_MODE, N_ITER = [13, 25], 6, 2

# The paper's SFINCS (drift-kinetic) benchmark for this configuration, verbatim
# from the Zenodo archive (calculations/figure16); <J.B> [T*A/m^2] on S_EVAL.
S_EVAL = np.linspace(0.02, 0.98, 49)
JDOTB_SFINCS = np.full(S_EVAL.shape, np.nan)  # QH SFINCS points are a PDF-only figure in the archive

# --------------------------- published deck, current erased -----------------
if not DECK.is_file():
    raise SystemExit(f"Zenodo dataset not found at {ZENODO}\n"
                     "set VMEX_ZENODO_2205_02914 to its root directory")
inp_pub = vj.VmecInput.from_file(DECK)      # boundary + pressure + spline current

def truncate_boundary(inp, max_mode):
    """Drop boundary/axis harmonics above ``max_mode`` (CI-budget resolution)."""
    if max_mode is None or max_mode + 1 >= inp.mpol:
        return inp
    m, nt = max_mode, inp.ntor
    rbc = np.zeros((2 * m + 1, m + 1)); zbs = np.zeros((2 * m + 1, m + 1))
    for n in range(-m, m + 1):
        rbc[n + m], zbs[n + m] = inp.rbc[n + nt, :m + 1], inp.zbs[n + nt, :m + 1]
    return dataclasses.replace(
        inp, mpol=m + 1, ntor=m, rbc=rbc, zbs=zbs, rbs=None, zbc=None, raxis_s=None,
        raxis_c=np.asarray(inp.raxis_c)[:m + 1], zaxis_s=np.asarray(inp.zaxis_s)[:m + 1],
        zaxis_c=None)

inp = dataclasses.replace(
    truncate_boundary(inp_pub, MAX_MODE),
    ns_array=NS_ARRAY, ftol_array=[1e-11] * len(NS_ARRAY),
    niter_array=[2000] + [4000] * (len(NS_ARRAY) - 1),
    # erase the published (already self-consistent) current profile:
    ncurr=1, pcurr_type="power_series", ac=np.concatenate([[1.0], np.zeros(20)]),
    curtor=0.0)

# --------------------------- Picard to self-consistency ---------------------
profiles = bs.KineticProfiles(ne_coeffs=N0 * np.array([1, 0, 0, 0, 0, -1.0]),
                              Te_coeffs=T0 * np.array([1, -1.0]),
                              Ti_coeffs=T0 * np.array([1, -1.0]))
res = bs.self_consistent_bootstrap(inp, profiles, HELICITY_N, n_iter=N_ITER, tol=TOL,
                                   s_eval=S_EVAL, verbose=True)
eq, f_boot = res.equilibrium, res.history[-1]["f_boot"]

# --------------------------- compare with the stored Zenodo curves ----------
wout_pub = vj.read_wout(WOUT_PUB)
jd_pub = np.interp(S_EVAL, np.linspace(0, 1, int(wout_pub.ns)), np.asarray(wout_pub.jdotb))
jv = np.interp(S_EVAL, np.linspace(0, 1, int(eq.wout.ns)), np.asarray(eq.wout.jdotb))
jr = np.asarray(bs.j_dot_B_redl(profiles, bs.redl_geometry_from_wout(eq.wout, S_EVAL), HELICITY_N)[0])
inner = (S_EVAL >= 0.1) & (S_EVAL <= 0.9)
rms = lambda a, b: float(np.sqrt(np.mean(((a[inner] - b[inner]) / b[inner]) ** 2)))  # noqa: E731
print(f"\n[{TAG}] converged = {res.converged} in {res.iterations} Picard iterations")
print(f"[{TAG}] final f_boot = {f_boot:.3e}")
print(f"[{TAG}] I_p = {res.input.curtor / 1e6:+.4f} MA (published CURTOR "
      f"{inp_pub.curtor / 1e6:+.4f} MA, rel {abs(res.input.curtor / inp_pub.curtor - 1):.3f})")
print(f"[{TAG}] <J.B> vs published Zenodo profile: {rms(jv, jd_pub):.4f} RMS")
print(f"[{TAG}] <J.B>_Redl vs SFINCS benchmark:    {rms(jr, JDOTB_SFINCS):.4f} RMS")

# --------------------------- outputs (deck, wout, curves, figures) ----------
import matplotlib; matplotlib.use("Agg")  # noqa: E401
import matplotlib.pyplot as plt
OUT_DIR.mkdir(parents=True, exist_ok=True)
res.input.to_indata(OUT_DIR / f"input.{TAG}_bootstrap_selfconsistent")
vj.write_wout(OUT_DIR / f"wout_{TAG}_bootstrap_selfconsistent.nc", eq.wout)
np.savez(OUT_DIR / f"bootstrap_curves_{TAG}.npz", s=S_EVAL, jv=jv, jr=jr,
         jd_pub=jd_pub, sfincs=JDOTB_SFINCS, f_boot=f_boot,
         curtor=res.input.curtor, curtor_pub=inp_pub.curtor, title=TITLE)

BLUE, AQUA, RUST, INK2, GRID = "#2a78d6", "#1baf7a", "#c95d38", "#52514e", "#e4e3e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "figure.facecolor": "white"})

def draw_panel(ax, d):
    """One config: <J.B>_vmec vs <J.B>_Redl after self-consistency + references."""
    ax.plot(d["s"], d["jd_pub"] / 1e6, color=INK2, lw=3.2, alpha=0.30,
            solid_capstyle="round", label="published equilibrium (Zenodo)")
    ax.plot(d["s"], d["jv"] / 1e6, color=BLUE, lw=1.7, label=r"$\langle J\cdot B\rangle$ VMEC (this run)")
    ax.plot(d["s"], d["jr"] / 1e6, "--", color=AQUA, lw=1.7, label=r"$\langle J\cdot B\rangle$ Redl (profiles)")
    if np.any(np.isfinite(d["sfincs"])):   # SFINCS points published for QA (fig 16); QH only as a PDF
        ax.plot(d["s"][1::3], d["sfincs"][1::3] / 1e6, "o", ms=3.6, mfc="none",
                mew=1.1, color=RUST, ls="none", label="SFINCS drift-kinetic (paper)")
    ax.annotate(f"$f_{{boot}}$ = {float(d['f_boot']):.1e}\n$I_p$ = {float(d['curtor']) / 1e6:+.3f} MA "
                f"(published {float(d['curtor_pub']) / 1e6:+.3f})",
                (0.03, 0.05), xycoords="axes fraction", fontsize=8, color=INK2)
    ax.set_title(str(d["title"]), fontsize=10, loc="left")
    ax.set_xlabel("normalized toroidal flux  s")
    ax.grid(True, color=GRID, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
draw_panel(ax, np.load(OUT_DIR / f"bootstrap_curves_{TAG}.npz"))
ax.set_ylabel(r"$\langle J\cdot B\rangle$  [MA T / m$^2$]")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT_DIR / f"{TAG}_bootstrap_jdotb.png")
print(f"wrote {OUT_DIR}/ (deck, wout, curves npz, {TAG}_bootstrap_jdotb.png)")

# combined README figure once both configurations have been run
paths = {t: Path(f"output_{t}_bootstrap_selfconsistent/bootstrap_curves_{t}.npz")
         for t in ("QA", "QH")}
if all(p.exists() for p in paths.values()):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=150, sharex=True)
    for ax, (t, p) in zip(axes, paths.items()):
        draw_panel(ax, np.load(p))
    axes[0].set_ylabel(r"$\langle J\cdot B\rangle$  [MA T / m$^2$]")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center")
    fig.tight_layout(); fig.savefig("readme_bootstrap.png")
    print("wrote readme_bootstrap.png (both configurations available)")
