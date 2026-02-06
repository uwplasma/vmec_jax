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
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar, vmec_fsq_from_tomnsps
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


CASES = [
    ("circular_tokamak", "examples/data/input.circular_tokamak", "examples/data/wout_circular_tokamak_reference.nc"),
    ("li383_low_res", "examples/data/input.li383_low_res", "examples/data/wout_li383_low_res_reference.nc"),
]


def main():
    enable_x64()

    try:
        import netCDF4  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("This example requires netCDF4 (pip install -e .[netcdf]).") from e

    outdir = REPO_ROOT / "examples/outputs"
    outdir.mkdir(exist_ok=True)

    for name, input_rel, wout_rel in CASES:
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
        k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)
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
        norms = vmec_force_norms_from_bcovar(bc=k.bc, trig=trig, wout=wout, s=static.s)
        scal = vmec_fsq_from_tomnsps(frzl=frzl, norms=norms, lconm1=bool(getattr(cfg, "lconm1", True)))

        print(f"== {name} ==")
        print(f"  ref: fsqr={wout.fsqr:.3e}  fsqz={wout.fsqz:.3e}  fsql={wout.fsql:.3e}")
        print(f"  jax: fsqr={scal.fsqr:.3e}  fsqz={scal.fsqz:.3e}  fsql={scal.fsql:.3e}")
        print(f"  rel: fsqr={_rel(scal.fsqr, wout.fsqr):.3e}  fsqz={_rel(scal.fsqz, wout.fsqz):.3e}  fsql={_rel(scal.fsql, wout.fsql):.3e}")

        np.savez(
            outdir / f"step10_getfsq_parity_{name}.npz",
            fsqr=float(scal.fsqr),
            fsqz=float(scal.fsqz),
            fsql=float(scal.fsql),
            fsqr_ref=float(wout.fsqr),
            fsqz_ref=float(wout.fsqz),
            fsql_ref=float(wout.fsql),
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
        )


if __name__ == "__main__":
    main()
