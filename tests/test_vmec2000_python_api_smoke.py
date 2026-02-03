from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


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


def test_vmec2000_python_api_produces_reference_wout_for_circular_tokamak(tmp_path: Path):
    pytest.importorskip("netCDF4")
    pytest.importorskip("mpi4py")

    # Import MPI early, before importing the VMEC2000 extension. On some macOS
    # setups the dynamic loader order matters for locating libmpi.
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
    input_path = repo_root / "examples/data/input.circular_tokamak"
    ref_wout_path = repo_root / "examples/data/wout_circular_tokamak_reference.nc"
    assert input_path.exists()
    assert ref_wout_path.exists()

    out_wout_path = _run_vmec2000_case(
        vmec=vmec,
        input_path=input_path,
        tmp_path=tmp_path,
        fcomm=MPI.COMM_WORLD.py2f(),
        barrier=MPI.COMM_WORLD.Barrier,
    )

    from netCDF4 import Dataset  # noqa: PLC0415

    def _arr(ds, name: str):
        return np.asarray(ds.variables[name][:])

    with Dataset(out_wout_path) as ds_new, Dataset(ref_wout_path) as ds_ref:
        # Match the VMEC2000 regression tests: a few high-signal fields.
        for field in ["iotaf", "rmnc", "zmns", "lmns", "bmnc"]:
            np.testing.assert_allclose(_arr(ds_new, field), _arr(ds_ref, field), atol=1e-10, rtol=1e-6)
