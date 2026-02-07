from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


CASES = [
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


def _solve_metric(*, input_path: Path, max_iter: int, gn_damping: float, gn_cg_tol: float, gn_cg_maxiter: int) -> float:
    """Return final fsq = fsqr+fsqz+fsql from a vmec_gn fixed-boundary solve."""
    run = run_fixed_boundary(
        input_path,
        solver="vmec_gn",
        max_iter=int(max_iter),
        step_size=1.0,
        gn_damping=float(gn_damping),
        gn_cg_tol=float(gn_cg_tol),
        gn_cg_maxiter=int(gn_cg_maxiter),
        verbose=False,
    )
    res = run.result
    fsqr2 = float(getattr(res, "fsqr2_history")[-1])
    fsqz2 = float(getattr(res, "fsqz2_history")[-1])
    fsql2 = float(getattr(res, "fsql2_history")[-1])
    return fsqr2 + fsqz2 + fsql2


def main():
    enable_x64()

    try:
        import netCDF4  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("This example requires netCDF4 (pip install -e .[netcdf]).") from e

    outdir = REPO_ROOT / "examples/outputs"
    outdir.mkdir(exist_ok=True)

    solve_metric = "--solve-metric" in sys.argv
    include_all = "--all" in sys.argv
    # For `--solve-metric`, default to a tiny axisymmetric subset so the script
    # stays interactive (avoids multi-minute JAX recompiles across many shapes).
    solve_default_cases = {"shaped_tokamak_pressure", "circular_tokamak"}
    cases = CASES
    if solve_metric and not include_all:
        cases = [c for c in CASES if c[0] in solve_default_cases]
    # Default settings: aggressive (better convergence), but bounded.
    gn_damping = 1.0e-6
    gn_cg_tol = 1.0e-10
    gn_cg_maxiter = 200
    max_iter_axisym = 30
    max_iter_3d = 10

    for name, input_rel, wout_rel in cases:
        input_path = REPO_ROOT / input_rel
        wout_path = REPO_ROOT / wout_rel
        cfg, indata = load_config(str(input_path))
        wout = read_wout(wout_path)

        # For true output-parity comparisons against the bundled `wout` files,
        # use the same (ntheta,nzeta) grid VMEC used to compute `fsqr/fsqz/fsql`.
        grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
        static = build_static(cfg, grid=grid)
        trig = vmec_trig_tables(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(wout.nfp),
            mmax=int(wout.mpol) - 1,
            nmax=int(wout.ntor),
            lasym=bool(wout.lasym),
        )

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

        if solve_metric:
            # Keep 3D runs short (this is a progress metric, not a benchmark).
            max_iter = max_iter_axisym if int(cfg.ntor) == 0 else max_iter_3d
            fsq_final = _solve_metric(
                input_path=input_path,
                max_iter=max_iter,
                gn_damping=gn_damping,
                gn_cg_tol=gn_cg_tol,
                gn_cg_maxiter=gn_cg_maxiter,
            )
            print(f"  solve: solver=vmec_gn max_iter={max_iter} fsq_final={fsq_final:.3e}")

        np.savez(
            outdir / f"step10_getfsq_parity_{name}.npz",
            fsqr=float(fsqr),
            fsqz=float(fsqz),
            fsql=float(fsql),
            fsqr_ref=float(wout.fsqr),
            fsqz_ref=float(wout.fsqz),
            fsql_ref=float(wout.fsql),
            solve_metric=bool(solve_metric),
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
        )


if __name__ == "__main__":
    main()
