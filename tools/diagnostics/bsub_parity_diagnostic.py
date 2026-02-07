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
from vmec_jax.geom import eval_geom
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.wout import read_wout, state_from_wout


CASES = [
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


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


def _half_mesh_coeffs(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    if a.shape[0] > 1:
        out[1:] = 0.5 * (a[1:] + a[:-1])
    return out


def main():
    enable_x64()

    try:
        import netCDF4  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit("This example requires netCDF4 (pip install -e .[netcdf]).") from e

    for name, input_rel, wout_rel in CASES:
        input_path = REPO_ROOT / input_rel
        wout_path = REPO_ROOT / wout_rel
        cfg, _ = load_config(str(input_path))
        wout = read_wout(wout_path)

        # VMEC internal angle grid for parity diagnostics.
        grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
        static = build_static(cfg, grid=grid)
        st = state_from_wout(wout)

        bc = vmec_bcovar_half_mesh_from_wout(state=st, static=static, wout=wout, use_wout_bsup=False, use_vmec_synthesis=True)

        # Geometry on the half mesh for metric comparison.
        st_half = replace(
            st,
            Rcos=_half_mesh_coeffs(np.asarray(st.Rcos)),
            Rsin=_half_mesh_coeffs(np.asarray(st.Rsin)),
            Zcos=_half_mesh_coeffs(np.asarray(st.Zcos)),
            Zsin=_half_mesh_coeffs(np.asarray(st.Zsin)),
            Lcos=np.asarray(st.Lcos),
            Lsin=np.asarray(st.Lsin),
        )
        geom = eval_geom(st_half, static)

        # Nyquist basis for reference fields.
        modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
        basis_nyq = build_helical_basis(modes_nyq, AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp))
        bsupu_ref = np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
        bsupv_ref = np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
        bsubu_ref = np.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
        bsubv_ref = np.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))
        bmag_ref = np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))

        # Exclude axis surface.
        sl = slice(1, None)

        bsupu_err = _rel_rms(np.asarray(bc.bsupu)[sl], bsupu_ref[sl])
        bsupv_err = _rel_rms(np.asarray(bc.bsupv)[sl], bsupv_ref[sl])
        bsubu_err = _rel_rms(np.asarray(bc.bsubu)[sl], bsubu_ref[sl])
        bsubv_err = _rel_rms(np.asarray(bc.bsubv)[sl], bsubv_ref[sl])

        guu_err = _rel_rms(np.asarray(bc.guu)[sl], np.asarray(geom.g_tt)[sl])
        guv_err = _rel_rms(np.asarray(bc.guv)[sl], np.asarray(geom.g_tp)[sl])
        gvv_err = _rel_rms(np.asarray(bc.gvv)[sl], np.asarray(geom.g_pp)[sl])

        b2_dot = (np.asarray(bc.bsupu) * np.asarray(bc.bsubu) + np.asarray(bc.bsupv) * np.asarray(bc.bsubv))[sl]
        b2_bmag = (bmag_ref**2)[sl]
        b2_err = _rel_rms(b2_dot, b2_bmag)

        print(f"== {name} ==")
        print(f"  bsupu rel RMS: {bsupu_err:.3e}  bsupv rel RMS: {bsupv_err:.3e}")
        print(f"  bsubu rel RMS: {bsubu_err:.3e}  bsubv rel RMS: {bsubv_err:.3e}")
        print(f"  metric rel RMS: guu {guu_err:.3e}  guv {guv_err:.3e}  gvv {gvv_err:.3e}")
        print(f"  BdotB vs |B|^2 rel RMS: {b2_err:.3e}")


if __name__ == "__main__":
    main()
