from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar, vmec_fsq_from_tomnsps
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout


def _vmec2000_root() -> Path | None:
    # Expected layout in this monorepo-style workspace:
    #   <workspace>/vmec_jax_git
    #   <workspace>/VMEC2000
    root = Path(__file__).resolve().parents[2]
    cand = root / "VMEC2000"
    return cand if cand.exists() else None


def _add_vmec2000_python_to_syspath(vmec2000_root: Path) -> None:
    # Prefer the scikit-build install tree if present.
    matches = sorted(vmec2000_root.glob("_skbuild/*/cmake-install/python"))
    if not matches:
        raise FileNotFoundError("VMEC2000 python install tree not found under _skbuild/*/cmake-install/python")

    import sys

    sys.path.insert(0, str(matches[0]))


def _run_vmec2000_case(*, vmec, input_path: Path, tmp_path: Path, fcomm: int, barrier) -> Path:
    # Flags used by VMEC2000 python wrapper `runvmec`:
    restart_flag = 1
    readin_flag = 2
    timestep_flag = 4
    output_flag = 8
    reset_jacdt_flag = 32

    ictrl = np.zeros(5, dtype=np.int32)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Read input.
        ictrl[:] = 0
        ictrl[0] = restart_flag + readin_flag
        vmec.runvmec(ictrl, str(input_path), False, fcomm, "")
        assert int(ictrl[1]) == 0
        vmec.cleanup(False)

        # Force VMEC to recompute its axis guess.
        vmec.vmec_input.raxis_cc = 0
        vmec.vmec_input.raxis_cs = 0
        vmec.vmec_input.zaxis_cc = 0
        vmec.vmec_input.zaxis_cs = 0
        vmec.reinit()

        # Timestep + output.
        ictrl[:] = 0
        ictrl[0] = restart_flag + reset_jacdt_flag + timestep_flag + output_flag
        vmec.runvmec(ictrl, str(input_path), False, fcomm, "")

        # 11 = successful_term_flag in `vmec_params.f`.
        assert int(ictrl[1]) == 11
        barrier()

        outs = list(Path(tmp_path).glob("wout_*.nc"))
        assert outs, "VMEC2000 did not produce a wout_*.nc file"
        out = outs[0]
    finally:
        vmec.cleanup(True)
        os.chdir(cwd)

    return out


@pytest.mark.vmec2000
def test_vmec2000_step10_scalars_match_vmec_jax_for_circular_tokamak(tmp_path: Path):
    """Integration parity: VMEC2000 run -> wout.fsq* vs vmec_jax Step-10 scalars.

    This is skipped by default since running VMEC2000 in CI requires a Fortran+MPI
    toolchain and can be slow. Enable with:
        VMEC2000_INTEGRATION=1 pytest -q -m vmec2000
    """
    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 integration parity tests")

    pytest.importorskip("netCDF4")
    pytest.importorskip("mpi4py")

    try:
        from mpi4py import MPI  # noqa: PLC0415
    except Exception as e:
        pytest.skip(f"mpi4py MPI backend not available: {e!r}")

    vmec2000_root = _vmec2000_root()
    if vmec2000_root is None:
        pytest.skip("VMEC2000 checkout not found next to vmec_jax_git")

    try:
        _add_vmec2000_python_to_syspath(vmec2000_root)
    except FileNotFoundError:
        pytest.skip("VMEC2000 python extension not built (missing _skbuild/*/cmake-install/python)")

    import vmec  # type: ignore  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples/input.circular_tokamak"
    assert input_path.exists()

    out_wout_path = _run_vmec2000_case(
        vmec=vmec,
        input_path=input_path,
        tmp_path=tmp_path,
        fcomm=MPI.COMM_WORLD.py2f(),
        barrier=MPI.COMM_WORLD.Barrier,
    )

    cfg, indata = load_config(str(input_path))
    wout = read_wout(out_wout_path)

    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)
    st = state_from_wout(wout)

    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
    )

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

    assert np.isfinite(scal.fsqr)
    assert np.isfinite(scal.fsqz)
    assert np.isfinite(scal.fsql)

    denom_r = max(abs(float(wout.fsqr)), 1e-20)
    denom_z = max(abs(float(wout.fsqz)), 1e-20)
    denom_l = max(abs(float(wout.fsql)), 1e-20)

    # Keep tolerances modest during parity push; tighten as residual conventions converge.
    assert abs(scal.fsqr - float(wout.fsqr)) / denom_r < 0.10
    assert abs(scal.fsqz - float(wout.fsqz)) / denom_z < 0.10
    assert abs(scal.fsql - float(wout.fsql)) / denom_l < 0.10

