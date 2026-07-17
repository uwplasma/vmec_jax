"""Cheap error/branch coverage for pure host-side paths (no solves).

These validation and error-handling branches are hard to reach from the
solver-driven tests but are important zero-crash guarantees; testing them
directly keeps the coverage gate meaningful without expensive solves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmex.core import optimize as opt
from vmex.core.errors import MgridNotFoundError
from vmex.core.input import VmecInput
from vmex.core.mgrid import MgridData, read_mgrid

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


def _valid_mgrid_kwargs(nextcur=1, ir=4, jz=5, kp=3):
    shape = (nextcur, kp, jz, ir)
    return dict(
        rmin=0.5, rmax=1.5, zmin=-0.5, zmax=0.5, ir=ir, jz=jz, kp=kp, nfp=1,
        nextcur=nextcur, mgrid_mode="N",
        coil_groups=tuple(f"g{i}" for i in range(nextcur)),
        raw_coil_cur=tuple(1.0 for _ in range(nextcur)),
        br=np.zeros(shape), bp=np.zeros(shape), bz=np.zeros(shape),
    )


def test_mgrid_data_valid_constructs():
    MgridData(**_valid_mgrid_kwargs())  # no exception


@pytest.mark.parametrize("field", ["br", "bp", "bz"])
def test_mgrid_data_rejects_wrong_field_shape(field):
    kwargs = _valid_mgrid_kwargs()
    kwargs[field] = np.zeros((1, 2, 3, 4))  # wrong shape
    with pytest.raises(ValueError, match=f"mgrid {field} shape"):
        MgridData(**kwargs)


def test_mgrid_data_rejects_coil_groups_length_mismatch():
    kwargs = _valid_mgrid_kwargs(nextcur=2)
    kwargs["coil_groups"] = ("only_one",)
    with pytest.raises(ValueError, match="coil_groups length"):
        MgridData(**kwargs)


def test_mgrid_data_rejects_raw_coil_cur_length_mismatch():
    kwargs = _valid_mgrid_kwargs(nextcur=2)
    kwargs["raw_coil_cur"] = (1.0,)
    with pytest.raises(ValueError, match="raw_coil_cur length"):
        MgridData(**kwargs)


def test_read_mgrid_missing_file_raises_typed_error(tmp_path):
    with pytest.raises(MgridNotFoundError):
        read_mgrid(tmp_path / "does_not_exist.nc")


def test_read_mgrid_non_netcdf_raises_typed_error(tmp_path):
    junk = tmp_path / "not_a_real.nc"
    junk.write_bytes(b"this is not netCDF at all")
    with pytest.raises(MgridNotFoundError):
        read_mgrid(junk)


# -- optimize.py solve-free branches --------------------------------------


def test_least_squares_x0_with_schedule_is_rejected():
    inp = VmecInput.from_file(DATA / "input.minimal_seed_nfp2")
    terms = [(opt.aspect_ratio, 5.0, 1.0)]
    with pytest.raises(ValueError, match="x0 cannot be combined"):
        opt.least_squares(terms, inp, max_mode=[1, 2], x0=np.zeros(3))


def test_boundary_pack_unpack_round_trip():
    inp = VmecInput.from_file(DATA / "input.minimal_seed_nfp2")
    x = opt.pack_boundary(inp, max_mode=2)
    names = opt.boundary_dof_names(inp, max_mode=2)
    assert len(names) == len(x) and all(isinstance(n, str) for n in names)
    # perturb, unpack, repack -> exact recovery of the modified vector.
    x2 = x + 1e-3
    inp2 = opt.unpack_boundary(inp, x2, max_mode=2)
    np.testing.assert_allclose(opt.pack_boundary(inp2, max_mode=2), x2, rtol=0, atol=1e-14)
    # RBC(0,0) (major radius) is excluded from the dof vector.
    assert "RBC(0,0)" not in names
