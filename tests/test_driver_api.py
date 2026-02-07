from __future__ import annotations

from pathlib import Path

import pytest

from vmec_jax.driver import example_paths, load_example, run_fixed_boundary, save_npz
from vmec_jax.vmec_tomnsp import vmec_angle_grid


def test_example_paths_and_load_example():
    pytest.importorskip("netCDF4")

    input_path, wout_path = example_paths("n3are_R7.75B5.7_lowres", root=Path(__file__).resolve().parents[1])
    assert input_path.exists()
    assert wout_path is not None and wout_path.exists()

    ex = load_example("n3are_R7.75B5.7_lowres", root=Path(__file__).resolve().parents[1], with_wout=True)
    assert ex.cfg.ns > 0
    assert ex.wout is not None
    assert ex.state is not None


def test_save_npz(tmp_path):
    path = save_npz(tmp_path / "demo.npz", a=[1, 2, 3], b=[4, 5, 6])
    assert path.exists()


def test_run_fixed_boundary_initial_guess():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.circular_tokamak"
    # Keep CI fast: use a small VMEC grid.
    grid = vmec_angle_grid(ntheta=12, nzeta=6, nfp=1, lasym=False)
    run = run_fixed_boundary(
        input_path,
        max_iter=1,
        use_initial_guess=True,
        vmec_project=False,
        verbose=False,
        grid=grid,
    )
    assert run.cfg.ns > 0
    assert run.state is not None
    assert run.result is None
