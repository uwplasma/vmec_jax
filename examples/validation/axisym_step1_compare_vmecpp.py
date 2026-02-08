#!/usr/bin/env python3
"""Compare VMEC++ vs vmec_jax first-step diagnostics (axisymmetric cases)."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np

import vmec_jax as vj
from vmec_jax.diagnostics import summarize_many
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.field import half_mesh_avg_from_full_mesh
from vmec_jax.profiles import eval_profiles
from vmec_jax.solve import vmecpp_first_step_diagnostics
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout
from vmec_jax.vmec_tomnsp import vmec_angle_grid, vmec_trig_tables
from vmec_jax.vmecpp_preconditioner import (
    _compute_preconditioning_matrix,
    _sm_sp_from_profiles,
    _sqrt_profiles_from_s,
    vmecpp_rz_preconditioner_matrices,
    vmecpp_wint_from_config,
)
from vmec_jax import vmec_residue


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.sqrt(np.mean(b * b))
    if denom == 0.0:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def _top_mode_rows(a: np.ndarray, *, k: int = 6):
    if a.ndim != 2:
        return []
    flat = a.reshape(-1)
    if flat.size == 0:
        return []
    idx = np.argsort(flat)[-k:][::-1]
    mpol, nrange = a.shape
    rows = []
    for ii in idx:
        m = int(ii // nrange)
        n = int(ii % nrange)
        rows.append((m, n, float(a[m, n])))
    return rows


def _maybe_print_array_diff(name: str, a: np.ndarray, b: np.ndarray):
    if a.size == 0 or b.size == 0:
        print(f"  {name}: skip empty jax={a.shape} vmecpp={b.shape}")
        return
    if a.shape != b.shape:
        print(f"  {name}: shape mismatch jax={a.shape} vmecpp={b.shape}")
        if a.shape[0] == b.shape[0] + 1:
            a = a[1:]
        min_ns = min(a.shape[0], b.shape[0])
        a = a[:min_ns]
        b = b[:min_ns]
    print(f"  {name}: rel_rms={_rel_rms(a, b):.3e}")


def _is_indata(path: Path) -> bool:
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            return stripped == "&INDATA"
    return False


def _find_indata2json(vmec_jax_root: Path) -> Path:
    env = os.getenv("VMECPP_INDATA2JSON")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    vmecpp_root = vmec_jax_root.parent / "vmecpp"
    candidates.extend(
        [
            vmecpp_root / "build311/_deps/indata2json-build/indata2json",
            vmecpp_root / "build2/_deps/indata2json-build/indata2json",
        ]
    )
    candidates.extend(vmecpp_root.glob("build*/_deps/indata2json-build/indata2json"))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "indata2json executable not found. Set VMECPP_INDATA2JSON or build vmecpp."
    )


def _ensure_vmecpp_input(input_path: Path, vmec_jax_root: Path) -> Path:
    if not _is_indata(input_path):
        return input_path
    indata2json = _find_indata2json(vmec_jax_root)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        local_input = tmpdir_path / input_path.name
        shutil.copyfile(input_path, local_input)
        subprocess.run(
            [str(indata2json), local_input.name],
            check=True,
            cwd=tmpdir_path,
        )
        json_name = input_path.name.replace("input.", "") + ".json"
        json_path = tmpdir_path / json_name
        if not json_path.is_file():
            raise RuntimeError(f"indata2json did not produce {json_path}")
        out_path = input_path.with_suffix(".json")
        shutil.copyfile(json_path, out_path)
        return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="circular_tokamak",
        choices=[
            "circular_tokamak",
            "shaped_tokamak_pressure",
            "vmecpp_solovev",
        ],
    )
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--vmecpp-precond", action="store_true")
    args = parser.parse_args()

    try:
        import vmecpp
    except Exception as exc:
        raise SystemExit(f"vmecpp import failed: {exc}")

    vmec_jax_root = Path(__file__).resolve().parents[2]
    data_dir = vmec_jax_root / "examples" / "data"
    input_path = data_dir / f"input.{args.case}"

    _, indata0 = vj.load_input(input_path)
    ns_array = indata0.get("NS_ARRAY", 0)
    ns_override = ns_array[0] if isinstance(ns_array, list) and ns_array else None
    run = vj.run_fixed_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        ns_override=ns_override,
    )
    signgs = int(run.signgs)
    cfg = run.static.cfg
    grid_vmec = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static_vmec = build_static(cfg, grid=grid_vmec)
    s = np.asarray(static_vmec.s, dtype=float)
    flux = flux_profiles_from_indata(run.indata, s, signgs=signgs)
    chipf_wout = half_mesh_avg_from_full_mesh(np.asarray(flux.chipf))
    phips = np.asarray(flux.phips).copy()
    if phips.shape[0] >= 1:
        phips[0] = 0.0
    pres = np.asarray(eval_profiles(run.indata, s).get("pressure", np.zeros_like(s)))

    class _WoutLikeVmec:
        def __init__(self):
            self.nfp = int(cfg.nfp)
            self.mpol = int(cfg.mpol)
            self.ntor = int(cfg.ntor)
            self.lasym = bool(cfg.lasym)
            self.signgs = int(signgs)
            self.phipf = np.asarray(flux.phipf)
            self.phips = phips
            self.chipf = chipf_wout
            self.pres = pres

    wout_like = _WoutLikeVmec()
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout_like.nfp),
        mmax=int(wout_like.mpol) - 1,
        nmax=int(wout_like.ntor),
        lasym=bool(wout_like.lasym),
    )
    k = vmec_forces_rz_from_wout(
        state=run.state,
        static=static_vmec,
        wout=wout_like,
        indata=None,
        use_vmec_synthesis=True,
        trig=trig,
    )
    diag_jax = vmecpp_first_step_diagnostics(
        run.state,
        run.static,
        indata=run.indata,
        signgs=int(run.signgs),
        step_size=args.step_size,
        include_edge=True,
        zero_m1=True,
        use_vmecpp_precond=bool(args.vmecpp_precond),
    )

    vmec_input_path = _ensure_vmecpp_input(input_path, vmec_jax_root)
    vmec_indata = vmecpp.cpp._vmecpp.VmecINDATA.from_file(vmec_input_path)
    diag_cpp = vmecpp.cpp._vmecpp.first_step_diagnostics(
        vmec_indata, max_threads=1, verbose=False
    )
    geom_cpp = vmecpp.cpp._vmecpp.first_step_geometry(
        vmec_indata, max_threads=1, verbose=False
    )
    prec_cpp = vmecpp.cpp._vmecpp.first_step_preconditioner(
        vmec_indata, max_threads=1, verbose=False
    )
    prec_comp_cpp = vmecpp.cpp._vmecpp.first_step_preconditioner_components(
        vmec_indata, max_threads=1, verbose=False
    )
    prec_inputs_cpp = vmecpp.cpp._vmecpp.first_step_preconditioner_inputs(
        vmec_indata, max_threads=1, verbose=False
    )

    print(
        f"[axisym_step1_compare] case={args.case} input={input_path} "
        f"(indata ntheta={vmec_indata.ntheta} nzeta={vmec_indata.nzeta} lasym={vmec_indata.lasym})"
    )
    for key in (
        "fsqr",
        "fsqz",
        "fsql",
        "fsqr1",
        "fsqz1",
        "fsql1",
        "f_norm1",
        "f_norm_rz",
        "f_norm_l",
    ):
        j = float(diag_jax[key])
        c = float(diag_cpp[key])
        rel = abs(j - c) / max(abs(c), 1e-30)
        print(f"  {key}: jax={j:.6e} vmecpp={c:.6e} rel={rel:.3e}")

    extra_keys = ("gcr2_raw", "gcz2_raw", "gcl2_raw", "rz_norm")
    for key in extra_keys:
        if key in diag_cpp and key in diag_jax:
            j = float(diag_jax[key])
            c = float(diag_cpp[key])
            rel = abs(j - c) / max(abs(c), 1e-30)
            print(f"  {key}: jax={j:.6e} vmecpp={c:.6e} rel={rel:.3e}")
    if "rz_norm" in diag_jax and "f_norm1" in diag_cpp:
        j = float(diag_jax["rz_norm"])
        c = 1.0 / float(diag_cpp["f_norm1"]) if float(diag_cpp["f_norm1"]) != 0.0 else float("inf")
        rel = abs(j - c) / max(abs(c), 1e-30)
        print(f"  rz_norm_from_f_norm1: jax={j:.6e} vmecpp={c:.6e} rel={rel:.3e}")

    for name in ("frcc", "fzsc", "flsc"):
        j = np.asarray(diag_jax[f"{name}_u"])
        c = np.asarray(diag_cpp[name])
        _maybe_print_array_diff(name, j, c)

    for name, key in (
        ("frcc_u", "frcc_mode_rms"),
        ("fzsc_u", "fzsc_mode_rms"),
        ("flsc_u", "flsc_mode_rms"),
    ):
        if key not in diag_jax:
            continue
        rows = _top_mode_rows(np.asarray(diag_jax[key]))
        if rows:
            print(f"  top {name} modes (m,n,rms)")
            for m, n, val in rows:
                print(f"    m={m:2d} n={n:2d} rms={val:.6e}")

    summarize_many(
        [
            ("jax_frcc_u", diag_jax["frcc_u"]),
            ("cpp_frcc", diag_cpp["frcc"]),
            ("jax_fzsc_u", diag_jax["fzsc_u"]),
            ("cpp_fzsc", diag_cpp["fzsc"]),
            ("jax_flsc_u", diag_jax["flsc_u"]),
            ("cpp_flsc", diag_cpp["flsc"]),
        ],
        indent="  ",
    )

    rcc, rss, zsc, zcs = vmec_residue.vmec_rz_decompose_signed(
        run.state,
        run.static,
        apply_scalxc=False,
        apply_basis_norm=True,
    )
    rcc = np.asarray(rcc)
    rss = np.asarray(rss)
    zsc = np.asarray(zsc)
    zcs = np.asarray(zcs)

    print("  decomposed geometry (raw)")
    _maybe_print_array_diff("rcc", rcc, np.asarray(geom_cpp["rcc"]))
    _maybe_print_array_diff("rss", rss, np.asarray(geom_cpp["rss"]))
    _maybe_print_array_diff("zsc", zsc, np.asarray(geom_cpp["zsc"]))
    _maybe_print_array_diff("zcs", zcs, np.asarray(geom_cpp["zcs"]))

    s = np.linspace(0.0, 1.0, rcc.shape[0], dtype=rcc.dtype)
    scalxc = np.asarray(vmec_residue.vmec_scalxc_from_s(s=s, mpol=rcc.shape[1]))[:, :, None]
    print("  decomposed geometry (scaled)")
    _maybe_print_array_diff("rcc", rcc * scalxc, np.asarray(geom_cpp["rcc"]))
    _maybe_print_array_diff("rss", rss * scalxc, np.asarray(geom_cpp["rss"]))
    _maybe_print_array_diff("zsc", zsc * scalxc, np.asarray(geom_cpp["zsc"]))
    _maybe_print_array_diff("zcs", zcs * scalxc, np.asarray(geom_cpp["zcs"]))

    rz_full = float(
        vmec_residue.vmec_rz_norm_from_state(
            state=run.state,
            static=run.static,
            apply_scalxc=True,
            apply_basis_norm=True,
            ns_min=0,
        )
    )
    rz_noaxis = float(
        vmec_residue.vmec_rz_norm_from_state(
            state=run.state,
            static=run.static,
            apply_scalxc=True,
            apply_basis_norm=True,
            ns_min=1,
        )
    )
    rz_raw = float(
        vmec_residue.vmec_rz_norm_from_state(
            state=run.state,
            static=run.static,
            apply_scalxc=False,
            apply_basis_norm=True,
            ns_min=0,
        )
    )
    rz_raw_noaxis = float(
        vmec_residue.vmec_rz_norm_from_state(
            state=run.state,
            static=run.static,
            apply_scalxc=False,
            apply_basis_norm=True,
            ns_min=1,
        )
    )
    print(f"  rz_norm (scaled, ns>=0)={rz_full:.6e}")
    print(f"  rz_norm (scaled, ns>=1)={rz_noaxis:.6e}")
    print(f"  rz_norm (raw, ns>=0)={rz_raw:.6e}")
    print(f"  rz_norm (raw, ns>=1)={rz_raw_noaxis:.6e}")

    rcc_u, rss_u, zsc_u, zcs_u = vmec_residue.vmec_rz_decompose_signed(
        run.state,
        run.static,
        apply_scalxc=False,
        apply_basis_norm=False,
    )
    rcc_u = np.asarray(rcc_u)
    rss_u = np.asarray(rss_u)
    zsc_u = np.asarray(zsc_u)
    zcs_u = np.asarray(zcs_u)
    print("  decomposed geometry (raw, unscaled basis)")
    _maybe_print_array_diff("rcc", rcc_u, np.asarray(geom_cpp["rcc"]))
    _maybe_print_array_diff("rss", rss_u, np.asarray(geom_cpp["rss"]))
    _maybe_print_array_diff("zsc", zsc_u, np.asarray(geom_cpp["zsc"]))
    _maybe_print_array_diff("zcs", zcs_u, np.asarray(geom_cpp["zcs"]))

    mats_jax, jmin_jax, jmax_jax = vmecpp_rz_preconditioner_matrices(
        bc=k.bc,
        k=k,
        trig=trig,
        s=np.asarray(static_vmec.s, dtype=float),
        cfg=static_vmec.cfg,
    )
    print("  preconditioner matrices (vmecpp vs jax)")
    for key in ("ar", "br", "dr", "az", "bz", "dz"):
        _maybe_print_array_diff(key, np.asarray(mats_jax[key]), np.asarray(prec_cpp[key]))
    print(f"  jmax: jax={jmax_jax} vmecpp={prec_cpp['jmax']}")
    if jmin_jax.size and len(prec_cpp["jmin"]) == jmin_jax.size:
        jmin_cpp = np.asarray(prec_cpp["jmin"], dtype=int).reshape(jmin_jax.shape)
        print(f"  jmin: rel_rms={_rel_rms(jmin_jax, jmin_cpp):.3e}")

    s_arr = np.asarray(static_vmec.s, dtype=float)
    sqrt_sf, sqrt_sh = _sqrt_profiles_from_s(s_arr)
    sm, sp = _sm_sp_from_profiles(sqrt_sf, sqrt_sh)
    w_int = vmecpp_wint_from_config(cfg=static_vmec.cfg)
    delta_s = float(s_arr[1] - s_arr[0]) if s_arr.size > 1 else 1.0
    ns_full = int(max(len(s_arr) - 1, 1))
    arm, ard, brm, brd, cxd = _compute_preconditioning_matrix(
        xs=np.asarray(k.bc.jac.zs, dtype=float)[1:],
        xu12=np.asarray(k.bc.jac.zu12, dtype=float)[1:],
        xu_e=np.asarray(k.pzu_even, dtype=float),
        xu_o=np.asarray(k.pzu_odd, dtype=float),
        x1_o=np.asarray(k.pz1_odd, dtype=float),
        r12=np.asarray(k.bc.jac.r12, dtype=float)[1:],
        total_pressure=np.asarray(k.bc.bsq, dtype=float)[1:],
        tau=np.asarray(k.bc.jac.tau, dtype=float)[1:],
        bsupv=np.asarray(k.bc.bsupv, dtype=float)[1:],
        sqrtg=np.asarray(k.bc.jac.sqrtg, dtype=float)[1:],
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns_full,
    )
    azm, azd, bzm, bzd, cxd_z = _compute_preconditioning_matrix(
        xs=np.asarray(k.bc.jac.rs, dtype=float)[1:],
        xu12=np.asarray(k.bc.jac.ru12, dtype=float)[1:],
        xu_e=np.asarray(k.pru_even, dtype=float),
        xu_o=np.asarray(k.pru_odd, dtype=float),
        x1_o=np.asarray(k.pr1_odd, dtype=float),
        r12=np.asarray(k.bc.jac.r12, dtype=float)[1:],
        total_pressure=np.asarray(k.bc.bsq, dtype=float)[1:],
        tau=np.asarray(k.bc.jac.tau, dtype=float)[1:],
        bsupv=np.asarray(k.bc.bsupv, dtype=float)[1:],
        sqrtg=np.asarray(k.bc.jac.sqrtg, dtype=float)[1:],
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns_full,
    )
    print("  preconditioner components (vmecpp vs jax)")
    _maybe_print_array_diff("arm", arm, np.asarray(prec_comp_cpp["arm"]))
    _maybe_print_array_diff("brm", brm, np.asarray(prec_comp_cpp["brm"]))
    _maybe_print_array_diff("azm", azm, np.asarray(prec_comp_cpp["azm"]))
    _maybe_print_array_diff("bzm", bzm, np.asarray(prec_comp_cpp["bzm"]))
    _maybe_print_array_diff("ard", ard, np.asarray(prec_comp_cpp["ard"]))
    _maybe_print_array_diff("brd", brd, np.asarray(prec_comp_cpp["brd"]))
    _maybe_print_array_diff("azd", azd, np.asarray(prec_comp_cpp["azd"]))
    _maybe_print_array_diff("bzd", bzd, np.asarray(prec_comp_cpp["bzd"]))
    _maybe_print_array_diff("cxd", cxd, np.asarray(prec_comp_cpp["cxd"]))
    _maybe_print_array_diff("cxd_z", cxd_z, np.asarray(prec_comp_cpp["cxd"]))


    print("  preconditioner inputs (vmecpp vs jax)")
    _maybe_print_array_diff("zs", np.asarray(k.bc.jac.zs, dtype=float)[1:], np.asarray(prec_inputs_cpp["zs"]))
    _maybe_print_array_diff("zu12", np.asarray(k.bc.jac.zu12, dtype=float)[1:], np.asarray(prec_inputs_cpp["zu12"]))
    _maybe_print_array_diff("zu_e", np.asarray(k.pzu_even, dtype=float), np.asarray(prec_inputs_cpp["zu_e"]))
    _maybe_print_array_diff("zu_o", np.asarray(k.pzu_odd, dtype=float), np.asarray(prec_inputs_cpp["zu_o"]))
    _maybe_print_array_diff("z1_o", np.asarray(k.pz1_odd, dtype=float), np.asarray(prec_inputs_cpp["z1_o"]))
    _maybe_print_array_diff("rs", np.asarray(k.bc.jac.rs, dtype=float)[1:], np.asarray(prec_inputs_cpp["rs"]))
    _maybe_print_array_diff("ru12", np.asarray(k.bc.jac.ru12, dtype=float)[1:], np.asarray(prec_inputs_cpp["ru12"]))
    _maybe_print_array_diff("ru_e", np.asarray(k.pru_even, dtype=float), np.asarray(prec_inputs_cpp["ru_e"]))
    _maybe_print_array_diff("ru_o", np.asarray(k.pru_odd, dtype=float), np.asarray(prec_inputs_cpp["ru_o"]))
    _maybe_print_array_diff("r1_o", np.asarray(k.pr1_odd, dtype=float), np.asarray(prec_inputs_cpp["r1_o"]))
    _maybe_print_array_diff("r12", np.asarray(k.bc.jac.r12, dtype=float)[1:], np.asarray(prec_inputs_cpp["r12"]))
    _maybe_print_array_diff("tau", np.asarray(k.bc.jac.tau, dtype=float)[1:], np.asarray(prec_inputs_cpp["tau"]))
    _maybe_print_array_diff("bsupv", np.asarray(k.bc.bsupv, dtype=float)[1:], np.asarray(prec_inputs_cpp["bsupv"]))
    _maybe_print_array_diff("sqrtg", np.asarray(k.bc.jac.sqrtg, dtype=float)[1:], np.asarray(prec_inputs_cpp["sqrtg"]))
    _maybe_print_array_diff("total_pressure", np.asarray(k.bc.bsq, dtype=float)[1:], np.asarray(prec_inputs_cpp["total_pressure"]))
    _maybe_print_array_diff("w_int", w_int, np.asarray(prec_inputs_cpp["w_int"]))
    _maybe_print_array_diff("sqrt_sh", sqrt_sh, np.asarray(prec_inputs_cpp["sqrt_sh"]))
    _maybe_print_array_diff("sm", sm, np.asarray(prec_inputs_cpp["sm"]))
    _maybe_print_array_diff("sp", sp, np.asarray(prec_inputs_cpp["sp"]))
    if "delta_s" in prec_inputs_cpp:
        print(f"  delta_s: jax={delta_s:.6e} vmecpp={float(prec_inputs_cpp['delta_s']):.6e}")
    print(
        f"  grid: jax ns={len(s_arr)} ntheta_eff={len(w_int)} nzeta={int(k.bc.guu.shape[2])}"
        f" vmecpp ns_full1={int(prec_inputs_cpp['ns_full1'])} ns_half={int(prec_inputs_cpp['ns_half'])}"
        f" ntheta_eff={len(np.asarray(prec_inputs_cpp['w_int']))} nzeta={int(prec_inputs_cpp['nzeta'])}"
    )
    if "ns_min_h" in prec_inputs_cpp and "ns_max_h" in prec_inputs_cpp:
        print(
            f"  vmecpp radial ranges: ns_min_h={int(prec_inputs_cpp['ns_min_h'])}"
            f" ns_max_h={int(prec_inputs_cpp['ns_max_h'])}"
            f" ns_min_f1={int(prec_inputs_cpp['ns_min_f1'])}"
            f" ns_max_f1={int(prec_inputs_cpp['ns_max_f1'])}"
        )
    print(
        f"  vmecpp sizes: ntheta={int(prec_inputs_cpp['ntheta'])} ntheta_even={int(prec_inputs_cpp['ntheta_even'])}"
        f" ntheta_reduced={int(prec_inputs_cpp['ntheta_reduced'])}"
    )
    print(
        f"  w_int jax head={w_int[:3]} vmecpp head={np.asarray(prec_inputs_cpp['w_int'])[:3]}"
    )
    print(
        f"  sqrt_sh jax head={sqrt_sh[:3]} vmecpp head={np.asarray(prec_inputs_cpp['sqrt_sh'])[:3]}"
    )
    print(f"  w_int vmecpp={np.asarray(prec_inputs_cpp['w_int'])}")
    print(f"  sqrt_sh vmecpp={np.asarray(prec_inputs_cpp['sqrt_sh'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
