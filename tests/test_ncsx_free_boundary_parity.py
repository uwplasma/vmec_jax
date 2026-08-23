"""Second free-boundary geometry family: NCSX c09r00 (nfp=3) vs VMEC2000.

The free-boundary adjoint program needs coupled-certificate evidence on two
independent 3-D geometry families; until now the only one with an mgrid was
CTH-like (nfp=5).  This module adds the NCSX c09r00 family — the li383-class
plasma whose fixed-boundary deck (``input.li383_low_res``) is already a golden
case — built from the public c09r00 modular-coil set.

Provenance (2026-08-23):

- Coils: ``BENCHMARKS/FIELDLINES_TEST/coils.NCSX`` from
  github.com/PrincetonUniversity/STELLOPT (added ``2080076f``, read from tree
  ``v6.5.0-42-g9177f58``), sha256
  ``3c429da06f4c062887a497a16e2d2bd10f0ecb0b8858c252698631f3853da428``:
  ten groups (ModA-C, PF1-6, TF) carrying the c09r00 currents.
- ``examples/data/mgrid_ncsx_c09r00_small.nc`` (fetched asset, see
  ``assets/manifest.json``): MAKEGRID ``xgrid`` from that
  file (same STELLOPT tree, gfortran 13.4.0; scaled mode ``S``, stellarator
  symmetric, R [0.75, 2.0] x Z [-0.8, 0.8] m, ir=jz=28, kp=24), 4.5 MB —
  committed like the CTH fixture.  ``kp`` equals the deck's ``NZETA`` so
  vmex's trilinear phi interpolation lands exactly on the planes VMEC2000
  samples directly.
- ``examples/data/input.ncsx_c09r00_free_lowres``: the published c09r00
  free-boundary input (``BENCHMARKS/DIAGNO_TEST/input.ncsx``) with only the
  resolution reduced to NS 9/15/25, MPOL=7, NTOR=6, FTOL 1e-6/1e-8/1e-10.

Reference run pinned in ``VMEC2000_DIGEST``: ``xvmec2000``
(STELLOPT ``v6.5.0-42-g9177f58``) TERMINATED NORMALLY, ``ier_flag 0``,
``fsqr 9.91e-11``, vacuum on at iteration 86, 612 iterations on the final
grid, 28 s wall.  vmex converges the same ladder to ``fsqr 9.87e-11`` with
vacuum on at the same iteration 86 and the same 611 final-grid iterations.
Measured parity (Apple Silicon CPU): wb 9.4e-9, aspect 1.3e-7, volume 3.9e-7,
betatotal 5.7e-7, b0 1.0e-7 relative; rmnc 4.8e-6 / zmns 2.2e-5 / bmnc 3.6e-6
scale-relative over all surfaces; iotaf 1.0e-5.  Gates carry 10-100x
platform headroom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmex.core.multigrid import solve_free_boundary_multigrid  # noqa: E402
from vmex.core.solver import resolution_from_input  # noqa: E402
from vmex.core.wout import wout_from_state  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
DECK = DATA / "input.ncsx_c09r00_free_lowres"
MGRID = DATA / "mgrid_ncsx_c09r00_small.nc"

pytestmark = pytest.mark.skipif(
    not MGRID.exists(),
    reason="mgrid_ncsx_c09r00_small.nc not fetched (tools/fetch_assets.py)",
)

#: wout scalars from the reference xvmec2000 run described in the module
#: docstring (deck FTOL reached, ier_flag 0).
VMEC2000_DIGEST = {
    "wb": 0.09297188899229081,
    "aspect": 4.469878968116116,
    "volume_p": 2.9620330123325473,
    "betatotal": 0.040850072239684336,
    "b0": 1.4384662422326966,
    "iotaf_axis": 0.36349702149502316,
    "iotaf_edge": 0.6556519387317133,
    "rmnc_bdy_rms": 0.15742260325342344,
    "zmns_bdy_rms": 0.052728395931541466,
}
RTOL = {
    "wb": 1e-5,
    "aspect": 2e-4, "volume_p": 2e-4, "betatotal": 2e-4, "b0": 2e-4,
    "rmnc_bdy_rms": 2e-4, "zmns_bdy_rms": 2e-4,
    "iotaf_axis": 1e-3, "iotaf_edge": 1e-3,
}


def test_mgrid_matches_the_published_c09r00_currents():
    """PR smoke: the committed fixture is self-consistent, no solve needed.

    The mgrid was generated in scaled mode from coils that bake in the
    c09r00 currents, and the deck's EXTCUR is the published current set —
    so ``raw_coil_cur`` must reproduce EXTCUR exactly.  The probe value
    pins the extcur-weighted trilinear interpolation on the phi=0
    stellarator-symmetry plane, where B_R vanishes identically.
    """
    inp = VmecInput.from_file(DECK)
    data = read_mgrid(MGRID)
    assert (data.ir, data.jz, data.kp, data.nfp, data.nextcur) == (28, 28, 24, 3, 10)
    assert data.mgrid_mode == "S"
    assert data.coil_groups == (
        "ModA", "ModB", "ModC", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6", "TF")
    # NZETA == kp keeps the two codes' phi sampling identical (see docstring).
    assert resolution_from_input(inp).nzeta == data.kp
    np.testing.assert_allclose(
        np.asarray(inp.extcur, dtype=float)[: data.nextcur],
        np.asarray(data.raw_coil_cur), rtol=1e-12)

    field = MgridField.from_mgrid_data(
        data, extcur=np.asarray(inp.extcur, dtype=float)[: data.nextcur])
    br, bp, bz = field.b_cyl(
        np.array([[1.4]]), np.array([[0.0]]), np.array([[0.0]]))
    assert abs(float(br[0, 0])) < 1e-12
    np.testing.assert_allclose(float(bp[0, 0]), 1.9666430906559516, rtol=1e-8)
    np.testing.assert_allclose(float(bz[0, 0]), 0.5525309587277026, rtol=1e-8)


@pytest.mark.full
def test_ncsx_free_ladder_matches_vmec2000():
    """Full NS 9/15/25 free-boundary ladder against the VMEC2000 digest.

    Measured 145 s cold with compilation, 55 s with a warm persistent
    compile cache (Apple Silicon CPU); vacuum turns on at iteration 86 in
    both codes and both stop at fsq < 1e-10 after 611 final-grid
    iterations.
    """
    inp = VmecInput.from_file(DECK)
    res = solve_free_boundary_multigrid(
        inp, mgrid_path=MGRID, raise_on_max_iterations=False)
    ftol = float(np.asarray(inp.ftol_array, dtype=float)[-1])
    assert res.converged, f"free boundary did not converge (fsqr={res.fsqr:.2e})"
    assert res.fsqr <= ftol and res.fsqz <= ftol and res.fsql <= ftol

    wout = wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged))
    got = {k: float(getattr(wout, k))
           for k in ("wb", "aspect", "volume_p", "betatotal", "b0")}
    iota = np.asarray(wout.iotaf, dtype=float)
    got["iotaf_axis"], got["iotaf_edge"] = float(iota[0]), float(iota[-1])
    got["rmnc_bdy_rms"] = float(np.sqrt(np.mean(np.asarray(wout.rmnc)[-1] ** 2)))
    got["zmns_bdy_rms"] = float(np.sqrt(np.mean(np.asarray(wout.zmns)[-1] ** 2)))

    problems = [
        f"{key}: vmex {got[key]:.9e} vs VMEC2000 {ref:.9e} "
        f"(rel {abs(got[key] / ref - 1.0):.2e} > {RTOL[key]:.0e})"
        for key, ref in VMEC2000_DIGEST.items()
        if not np.isclose(got[key], ref, rtol=RTOL[key], atol=0.0)
    ]
    assert not problems, "NCSX c09r00 parity:\n  " + "\n  ".join(problems)
