from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.diagnostics import print_stats
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import (
    rz_residual_coeffs_from_kernels,
    rz_residual_scalars_like_vmec,
    vmec_forces_rz_from_wout_reference_fields,
)
from vmec_jax.wout import read_wout, state_from_wout


def main():
    enable_x64()
    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples/input.circular_tokamak"
    wout_path = root / "examples/wout_circular_tokamak_reference.nc"

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)

    # Use a moderate grid; increase if you want finer diagnostics.
    cfg_hi = replace(cfg, ntheta=max(int(cfg.ntheta), 128), nzeta=max(int(cfg.nzeta), 128))
    static = build_static(cfg_hi)

    st = state_from_wout(wout)
    k = vmec_forces_rz_from_wout_reference_fields(state=st, static=static, wout=wout)
    coeffs = rz_residual_coeffs_from_kernels(k, static=static)
    scal = rz_residual_scalars_like_vmec(coeffs, bc=k.bc, wout=wout, s=static.s)

    print("== VMEC2000 reference scalars ==")
    print(f"fsqr={wout.fsqr:.3e}  fsqz={wout.fsqz:.3e}  fsql={wout.fsql:.3e}")
    print("== vmec_jax (R/Z force kernel; parity WIP) ==")
    print(f"fsqr_like={scal.fsqr_like:.3e}  fsqz_like={scal.fsqz_like:.3e}")

    gcr = np.asarray(coeffs.gcr_cos) + 1j * np.asarray(coeffs.gcr_sin)
    gcz = np.asarray(coeffs.gcz_cos) + 1j * np.asarray(coeffs.gcz_sin)
    print_stats("||gcr||_2 per-surface", np.linalg.norm(gcr, axis=1))
    print_stats("||gcz||_2 per-surface", np.linalg.norm(gcz, axis=1))

    out = root / "examples/outputs"
    out.mkdir(exist_ok=True)
    np.savez(
        out / "step10_forces_rz_kernel_report.npz",
        s=np.asarray(static.s),
        gcr_cos=np.asarray(coeffs.gcr_cos),
        gcr_sin=np.asarray(coeffs.gcr_sin),
        gcz_cos=np.asarray(coeffs.gcz_cos),
        gcz_sin=np.asarray(coeffs.gcz_sin),
        fsqr_like=float(scal.fsqr_like),
        fsqz_like=float(scal.fsqz_like),
        fsqr_ref=float(wout.fsqr),
        fsqz_ref=float(wout.fsqz),
    )
    print(f"Wrote {out / 'step10_forces_rz_kernel_report.npz'}")


if __name__ == \"__main__\":
    main()
