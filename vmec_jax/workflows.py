"""High-level workflows used by the minimal examples.

The goal of this module is to keep example scripts short and stable as the
internal parity work evolves.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import load_config
from .driver import run_fixed_boundary, write_wout_from_fixed_boundary_run
from .field import b_cartesian_from_bsup, bsup_from_geom
from .fieldlines import trace_fieldline_on_surface
from .geom import eval_geom
from .grids import AngleGrid
from .modes import ModeTable
from .plotting import (
    bmag_from_state_physical,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    surface_rz_from_wout_physical,
)
from .static import build_static
from .visualization import write_vtp_polyline, write_vts_structured_grid
from .wout import read_wout, state_from_wout


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 16)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 16)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _half_mesh(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else num


def axisym_showcase(
    *,
    outdir: str | Path | None = None,
    solver: str = "vmecpp_iter",
    max_iter: int = 30,
    step_size: float | None = None,
    plots: bool = True,
    verbose: bool = True,
):
    """Run bundled axisymmetric cases and write wouts + plots + parity summary."""
    root = _repo_root()
    data_dir = root / "examples" / "data"
    outdir = _ensure_dir(outdir or (root / "examples" / "outputs" / "axisym_showcase"))

    cases = [
        ("circular_tokamak", data_dir / "input.circular_tokamak", data_dir / "wout_circular_tokamak_reference.nc"),
        ("shaped_tokamak_pressure", data_dir / "input.shaped_tokamak_pressure", data_dir / "wout_shaped_tokamak_pressure_reference.nc"),
        ("vmecpp_solovev", data_dir / "input.vmecpp_solovev", data_dir / "wout_vmecpp_solovev_reference.nc"),
    ]

    if plots:
        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: F401
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    for name, input_path, wout_ref_path in cases:
        case_out = _ensure_dir(outdir / name)

        run_kws = dict(solver=str(solver), max_iter=int(max_iter), verbose=bool(verbose))
        if step_size is not None:
            run_kws["step_size"] = float(step_size)
        run = run_fixed_boundary(input_path, **run_kws)

        wout_new_path = case_out / f"wout_{name}_vmec_jax.nc"
        wout_new = write_wout_from_fixed_boundary_run(wout_new_path, run, include_fsq=True)
        wout_ref = read_wout(wout_ref_path)

        fsq_ref = float(wout_ref.fsqr + wout_ref.fsqz + wout_ref.fsql)
        fsq_new = float(wout_new.fsqr + wout_new.fsqz + wout_new.fsql)
        err_rmnc = _rel_rms(np.asarray(wout_new.rmnc), np.asarray(wout_ref.rmnc))
        err_zmns = _rel_rms(np.asarray(wout_new.zmns), np.asarray(wout_ref.zmns))
        pref = profiles_from_wout(wout_ref)
        pnew = profiles_from_wout(wout_new)
        err_iota = _rel_rms(pnew["iotaf"], pref["iotaf"])
        err_pres = _rel_rms(pnew["pres"], pref["pres"])

        print(f"\n== {name} ==")
        print(f"[vmec_jax] wrote:     {wout_new_path}")
        print(f"[vmec_jax] reference: {wout_ref_path}")
        print(f"[vmec_jax] fsq_total: ref={fsq_ref:.3e} new={fsq_new:.3e}")
        print(f"[vmec_jax] geom rms:  rmnc={err_rmnc:.3e} zmns={err_zmns:.3e}")
        print(f"[vmec_jax] prof rms:  iotaf={err_iota:.3e} pres={err_pres:.3e}")

        if not plots:
            continue

        import matplotlib.pyplot as plt

        theta = closed_theta_grid(256)
        phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        ns = int(wout_ref.ns)
        s_idx_list = [0, max(1, ns // 4), max(1, ns // 2), max(1, (3 * ns) // 4), ns - 1]
        st_ref = state_from_wout(wout_ref)
        st_new = state_from_wout(wout_new)
        indata = run.indata
        static = run.static

        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for s_idx in s_idx_list:
            R0, Z0 = surface_rz_from_wout_physical(wout_ref, theta=theta, phi=np.asarray([0.0]), s_index=int(s_idx), nyq=False)
            R1, Z1 = surface_rz_from_wout_physical(wout_new, theta=theta, phi=np.asarray([0.0]), s_index=int(s_idx), nyq=False)
            ax[0].plot(R0[:, 0], Z0[:, 0], lw=1.0)
            ax[1].plot(R1[:, 0], Z1[:, 0], lw=1.0)
        for a in ax:
            a.set_aspect("equal", "box")
            a.set_xlabel("R")
            a.set_ylabel("Z")
        ax[0].set_title("VMEC2000")
        ax[1].set_title("vmec_jax")
        fig.savefig(case_out / "surfaces_nested_phi0.png", dpi=180)
        plt.close(fig)

        B_ref = bmag_from_state_physical(st_ref, static, indata=indata, theta=theta, phi=phi, s_index=ns - 1)
        B_new = bmag_from_state_physical(st_new, static, indata=indata, theta=theta, phi=phi, s_index=ns - 1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        vmin = float(np.min([np.min(B_ref), np.min(B_new)]))
        vmax = float(np.max([np.max(B_ref), np.max(B_new)]))
        ax[0].imshow(B_ref, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax[1].imshow(B_new, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax[0].set_title("VMEC2000")
        ax[1].set_title("vmec_jax")
        fig.savefig(case_out / "bmag_lcfs.png", dpi=180)
        plt.close(fig)

        th3 = closed_theta_grid(120)
        ph3 = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        Rlcfs, Zlcfs = surface_rz_from_wout_physical(wout_new, theta=th3, phi=ph3, s_index=ns - 1, nyq=False)
        Blcfs = bmag_from_state_physical(st_new, static, indata=indata, theta=th3, phi=ph3, s_index=ns - 1)
        X = Rlcfs * np.cos(ph3[None, :])
        Y = Rlcfs * np.sin(ph3[None, :])
        fig = plt.figure(figsize=(6, 5), constrained_layout=True)
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.plot_surface(
            X,
            Y,
            Zlcfs,
            facecolors=plt.cm.viridis((Blcfs - Blcfs.min()) / max(Blcfs.ptp(), 1e-12)),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        fix_matplotlib_3d(ax3)
        fig.savefig(case_out / "lcfs_3d_bmag.png", dpi=180)
        plt.close(fig)

        print(f"[vmec_jax] plots:     {case_out}")

    print(f"\n[vmec_jax] done. outputs: {outdir}")


def write_axisym_overview(*, case: str = "circular_tokamak", outdir: str | Path | None = None):
    """Write a single VMEC-style overview panel for a bundled case."""
    root = _repo_root()
    outdir = _ensure_dir(outdir or (root / "docs" / "_static" / "figures"))
    ex = read_wout(root / "examples" / "data" / f"wout_{case}_reference.nc")
    st = state_from_wout(ex)
    cfg, indata = load_config(str(root / "examples" / "data" / f"input.{case}"))
    static = build_static(cfg)

    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    ns = int(ex.ns)
    s_index = ns - 1

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    for si in np.linspace(0, s_index, 9).round().astype(int):
        R, Z = surface_rz_from_wout_physical(ex, theta=theta, phi=np.asarray([0.0]), s_index=int(si), nyq=False)
        ax0.plot(R[:, 0], Z[:, 0], lw=1.2)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title("Nested surfaces (phi=0)")
    ax0.set_xlabel("R")
    ax0.set_ylabel("Z")

    B = bmag_from_state_physical(st, static, indata=indata, theta=theta, phi=phi, s_index=s_index)
    R3, Z3 = surface_rz_from_wout_physical(ex, theta=theta, phi=phi, s_index=s_index, nyq=False)
    PH = phi[None, :]
    X = R3 * np.cos(PH)
    Y = R3 * np.sin(PH)

    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=float(np.min(B)), vmax=float(np.max(B)))
    stride_t = 2
    stride_p = 2
    ax1.plot_surface(
        X[::stride_t, ::stride_p],
        Y[::stride_t, ::stride_p],
        Z3[::stride_t, ::stride_p],
        facecolors=cmap(norm(B))[::stride_t, ::stride_p],
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    fix_matplotlib_3d(ax1)
    ax1.set_title("LCFS 3D colored by |B|")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.pcolormesh(phi, theta, B, shading="auto", cmap="viridis")
    ax2.set_title("|B| on LCFS")
    ax2.set_xlabel("phi")
    ax2.set_ylabel("theta")

    ax3 = fig.add_subplot(gs[1, 1])
    sgrid = np.linspace(0.0, 1.0, int(ex.iotaf.size))
    ax3.plot(sgrid, np.asarray(ex.iotaf), lw=2, label="iota")
    ax3.plot(np.linspace(0.0, 1.0, int(ex.pres.size)), np.asarray(ex.pres), lw=2, label="pressure")
    ax3.legend(loc="best", frameon=True)
    ax3.set_title("Profiles")
    ax3.set_xlabel("s")

    outpath = Path(outdir) / f"{case}_overview.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print(f"[vmec_jax] wrote {outpath}")


def write_bsup_parity_figures(*, input_path: str | Path, wout_path: str | Path, outdir: str | Path):
    """Write bsup parity figures (VMEC2000 wout vs reconstructed fields)."""
    outdir = _ensure_dir(outdir)
    cfg, _ = load_config(str(input_path))
    wout = read_wout(str(wout_path))
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)

    st = state_from_wout(wout)
    st_half = replace(st, Rcos=_half_mesh(np.asarray(st.Rcos)), Rsin=_half_mesh(np.asarray(st.Rsin)), Zcos=_half_mesh(np.asarray(st.Zcos)), Zsin=_half_mesh(np.asarray(st.Zsin)))
    g = eval_geom(st_half, static)

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis = None
    from .fourier import build_helical_basis, eval_fourier

    basis = build_helical_basis(modes_nyq, grid)
    bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis))
    bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis))

    lamscale = float(wout.lamscale) if hasattr(wout, "lamscale") else 1.0
    bsupu, bsupv = bsup_from_geom(g, lamscale=lamscale)
    bsupu = np.asarray(bsupu)
    bsupv = np.asarray(bsupv)

    e_u = (bsupu - bsupu_ref) / (np.abs(bsupu_ref) + 1e-14)
    e_v = (bsupv - bsupv_ref) / (np.abs(bsupv_ref) + 1e-14)

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for tag, e in [("bsupu", e_u), ("bsupv", e_v)]:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        im = ax.imshow(e[-1].T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(f"rel. error {tag} (edge)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"{tag}_parity_error.png")
        plt.close(fig)


def write_bsub_parity_figures(*, input_path: str | Path, wout_path: str | Path, outdir: str | Path):
    """Write bsub parity figures (VMEC2000 wout vs metric reconstruction)."""
    outdir = _ensure_dir(outdir)
    cfg, _ = load_config(str(input_path))
    wout = read_wout(str(wout_path))
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)

    st = state_from_wout(wout)
    st_half = replace(st, Rcos=_half_mesh(np.asarray(st.Rcos)), Rsin=_half_mesh(np.asarray(st.Rsin)), Zcos=_half_mesh(np.asarray(st.Zcos)), Zsin=_half_mesh(np.asarray(st.Zsin)))
    g = eval_geom(st_half, static)

    from .fourier import build_helical_basis, eval_fourier

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis = build_helical_basis(modes_nyq, grid)
    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis))
    bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis))
    bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis))

    from .field import bsub_from_bsup

    bsubu, bsubv = bsub_from_bsup(g, bsupu, bsupv)
    bsubu = np.asarray(bsubu)
    bsubv = np.asarray(bsubv)

    e_u = (bsubu - bsubu_ref) / (np.abs(bsubu_ref) + 1e-14)
    e_v = (bsubv - bsubv_ref) / (np.abs(bsubv_ref) + 1e-14)

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for tag, e in [("bsubu", e_u), ("bsubv", e_v)]:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        im = ax.imshow(e[-1].T, origin="lower", aspect="auto", cmap="RdBu_r")
        ax.set_title(f"rel. error {tag} (edge)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"{tag}_parity_error.png")
        plt.close(fig)


def write_bmag_parity_figures(*, input_path: str | Path, wout_path: str | Path, outdir: str | Path):
    """Write |B| parity figure."""
    outdir = _ensure_dir(outdir)
    cfg, _ = load_config(str(input_path))
    wout = read_wout(str(wout_path))
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)

    st = state_from_wout(wout)
    st_half = replace(st, Rcos=_half_mesh(np.asarray(st.Rcos)), Rsin=_half_mesh(np.asarray(st.Rsin)), Zcos=_half_mesh(np.asarray(st.Zcos)), Zsin=_half_mesh(np.asarray(st.Zsin)))
    g = eval_geom(st_half, static)

    from .fourier import build_helical_basis, eval_fourier

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis = build_helical_basis(modes_nyq, grid)
    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis))
    B_ref = np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis))

    B2 = np.asarray(g.g_tt) * bsupu**2 + 2.0 * np.asarray(g.g_tp) * bsupu * bsupv + np.asarray(g.g_pp) * bsupv**2
    B = np.sqrt(np.maximum(B2, 0.0))
    e = (B - B_ref) / (np.abs(B_ref) + 1e-14)

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(e[-1].T, origin="lower", aspect="auto", cmap="RdBu_r")
    ax.set_title("rel. error |B| (edge)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "bmag_parity_error.png")
    plt.close(fig)


def export_vtk_surface_and_fieldline(
    *,
    input_path: str | Path,
    wout_path: str | Path | None = None,
    outdir: str | Path,
    s_index: int = -1,
    hi_res: bool = False,
    export_volume: bool = False,
    theta0: float = 0.0,
    phi0: float = 0.0,
    n_steps: int = 2000,
    dphi: float = 2e-3,
):
    """Export surface B data and one fieldline to VTK."""
    outdir = _ensure_dir(outdir)

    input_path = Path(input_path)
    cfg, _ = load_config(str(input_path))
    if wout_path is None:
        # Try a bundled reference wout based on input filename.
        inp = input_path.name
        case = inp[len("input.") :] if inp.startswith("input.") else inp
        candidate = _repo_root() / "examples" / "data" / f"wout_{case}_reference.nc"
        if not candidate.exists():
            raise FileNotFoundError("No --wout provided and no bundled reference wout found.")
        wout_path = candidate
    wout = read_wout(str(wout_path))
    if hi_res:
        cfg = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg)

    state = state_from_wout(wout)
    g = eval_geom(state, static)

    from .fourier import build_helical_basis, eval_fourier

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
    basis = build_helical_basis(modes_nyq, grid)
    bsupu = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis))
    bsupv = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis))

    B = b_cartesian_from_bsup(g, bsupu, bsupv, zeta=static.grid.zeta, nfp=cfg.nfp)
    B = np.asarray(B)
    Bmag = np.sqrt(np.sum(B**2, axis=-1))

    si = int(s_index)
    if si < 0:
        si = int(cfg.ns) - 1

    R = np.asarray(g.R[si])
    Z = np.asarray(g.Z[si])
    phi = np.asarray(static.grid.zeta) / cfg.nfp
    x = R * np.cos(phi)[None, :]
    y = R * np.sin(phi)[None, :]
    z = Z

    surface_path = Path(outdir) / f"surface_s{si:03d}.vts"
    write_vts_structured_grid(
        surface_path,
        x=x,
        y=y,
        z=z,
        point_data={"B": B[si], "Bmag": Bmag[si], "R": R, "Z": Z},
    )

    if export_volume:
        phi_v = np.asarray(static.grid.zeta) / cfg.nfp
        xv = np.asarray(g.R) * np.cos(phi_v)[None, None, :]
        yv = np.asarray(g.R) * np.sin(phi_v)[None, None, :]
        zv = np.asarray(g.Z)
        write_vts_structured_grid(Path(outdir) / "volume.vts", x=xv, y=yv, z=zv, point_data={"B": B, "Bmag": Bmag})

    fl = trace_fieldline_on_surface(
        R=R,
        Z=Z,
        bsupu=bsupu[si],
        bsupv=bsupv[si],
        Bmag=Bmag[si],
        nfp=cfg.nfp,
        theta0=float(theta0),
        phi0=float(phi0),
        n_steps=int(n_steps),
        dphi=float(dphi),
    )
    line_path = Path(outdir) / f"fieldline_s{si:03d}.vtp"
    write_vtp_polyline(line_path, points=np.stack([fl.x, fl.y, fl.z], axis=1), point_data={"Bmag": fl.Bmag})

    print(f"[vmec_jax] wrote {surface_path}")
    print(f"[vmec_jax] wrote {line_path}")


def step10_getfsq_parity_cases(*, root: str | Path | None = None, solve_metric: bool = False, include_all: bool = False):
    """Print Step-10 (fsqr/fsqz/fsql) parity vs bundled reference wouts.

    This is for diagnostics only; it writes small `.npz` summaries under
    `examples/outputs/`.
    """
    root = Path(root) if root is not None else _repo_root()
    outdir = _ensure_dir(root / "examples" / "outputs")

    # Keep `--solve-metric` interactive: default to a tiny subset unless `--all`.
    solve_default = {"shaped_tokamak_pressure", "circular_tokamak"}

    cases = [
        ("shaped_tokamak_pressure", "examples/data/input.shaped_tokamak_pressure", "examples/data/wout_shaped_tokamak_pressure_reference.nc"),
        ("circular_tokamak", "examples/data/input.circular_tokamak", "examples/data/wout_circular_tokamak_reference.nc"),
        ("li383_low_res", "examples/data/input.li383_low_res", "examples/data/wout_li383_low_res_reference.nc"),
        ("circular_tokamak_aspect_100", "examples/data/input.circular_tokamak_aspect_100", "examples/data/wout_circular_tokamak_aspect_100_reference.nc"),
        ("purely_toroidal_field", "examples/data/input.purely_toroidal_field", "examples/data/wout_purely_toroidal_field_reference.nc"),
        ("ITERModel", "examples/data/input.ITERModel", "examples/data/wout_ITERModel_reference.nc"),
        (
            "LandremanSengupta2019_section5.4_B2_A80",
            "examples/data/input.LandremanSengupta2019_section5.4_B2_A80",
            "examples/data/wout_LandremanSengupta2019_section5.4_B2_A80_reference.nc",
        ),
        ("n3are_R7.75B5.7_lowres", "examples/data/input.n3are_R7.75B5.7_lowres", "examples/data/wout_n3are_R7.75B5.7_lowres.nc"),
    ]

    if solve_metric and not include_all:
        cases = [c for c in cases if c[0] in solve_default]

    def _rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1e-300)

    from .static import build_static
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
    from .vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables

    for name, input_rel, wout_rel in cases:
        input_path = root / input_rel
        wout_path = root / wout_rel
        cfg, indata = load_config(str(input_path))
        wout = read_wout(str(wout_path))

        grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
        static = build_static(cfg, grid=grid)
        trig = vmec_trig_tables(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), mmax=int(wout.mpol) - 1, nmax=int(wout.ntor), lasym=bool(wout.lasym))

        st = state_from_wout(wout)
        k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata, use_wout_bsup=True)
        rzl = vmec_residual_internal_from_kernels(k, cfg_ntheta=int(cfg.ntheta), cfg_nzeta=int(cfg.nzeta), wout=wout, trig=trig)
        frzl = TomnspsRZL(
            frcc=rzl.frcc,
            frss=rzl.frss,
            fzsc=rzl.fzsc,
            fzcs=rzl.fzcs,
            flsc=rzl.flsc,
            flcs=rzl.flcs,
            frsc=rzl.frsc,
            frcs=rzl.frcs,
            fzcc=rzl.fzcc,
            fzss=rzl.fzss,
            flcc=rzl.flcc,
            flss=rzl.flss,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=static.s, signgs=int(wout.signgs))
        scal = vmec_fsq_from_tomnsps_dynamic(frzl=frzl, norms=norms, lconm1=bool(getattr(cfg, "lconm1", True)))
        fsqr = float(scal.fsqr)
        fsqz = float(scal.fsqz)
        fsql = float(scal.fsql)

        print(f"== {name} ==")
        print(f"  ref: fsqr={wout.fsqr:.3e}  fsqz={wout.fsqz:.3e}  fsql={wout.fsql:.3e}")
        print(f"  jax: fsqr={fsqr:.3e}  fsqz={fsqz:.3e}  fsql={fsql:.3e}")
        print(f"  rel: fsqr={_rel(fsqr, wout.fsqr):.3e}  fsqz={_rel(fsqz, wout.fsqz):.3e}  fsql={_rel(fsql, wout.fsql):.3e}")

        np.savez(outdir / f"step10_getfsq_parity_{name}.npz", fsqr=fsqr, fsqz=fsqz, fsql=fsql, fsqr_ref=float(wout.fsqr), fsqz_ref=float(wout.fsqz), fsql_ref=float(wout.fsql))
