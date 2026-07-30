"""Dimensional similarity transforms for inputs, mgrid fields, and WOUT."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vmex.core import cli
from vmex.core.errors import INPUT_ERROR_FLAG
from vmex.core.input import VmecInput
from vmex.core.mgrid import MgridData, read_mgrid, write_mgrid
from vmex.core.multigrid import solve_free_boundary_multigrid, solve_multigrid
from vmex.core.scaling import (
    aries_cs_scales,
    input_minor_radius,
    probe_input,
    scale_input,
    scale_mgrid,
    scale_wout,
)
from vmex.core.wout import read_wout, wout_from_state, write_wout

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_input_scaling_changes_only_dimensional_quantities():
    inp = VmecInput.from_file(DATA / "input.nfp2_QA_finite_beta")
    scaled = scale_input(inp, b_scale=2.0, r_scale=3.0)
    np.testing.assert_allclose(scaled.rbc, 3.0 * inp.rbc)
    np.testing.assert_allclose(scaled.raxis_c, 3.0 * inp.raxis_c)
    assert scaled.phiedge == 18.0 * inp.phiedge
    assert scaled.pres_scale == 4.0 * inp.pres_scale
    assert scaled.curtor == 6.0 * inp.curtor
    for name in ("am", "am_aux_s", "am_aux_f", "ac", "ai", "aphi"):
        np.testing.assert_array_equal(getattr(scaled, name), getattr(inp, name))

    prescribed_iota = dataclasses.replace(inp, ncurr=0)
    assert scale_input(
        prescribed_iota, b_scale=2.0, r_scale=3.0
    ).curtor == prescribed_iota.curtor
    with pytest.raises(ValueError, match="finite and positive"):
        scale_input(inp, b_scale=0.0)


def test_input_minor_radius_uses_vmec_boundary_convention():
    inp = VmecInput(
        mpol=2,
        ntor=0,
        rbc=np.array([[4.0, 0.7]]),
        zbs=np.array([[0.0, 0.7]]),
    )
    assert input_minor_radius(inp) == pytest.approx(0.7, abs=1e-14)
    assert input_minor_radius(scale_input(inp, r_scale=2.5)) == pytest.approx(
        1.75, abs=1e-14
    )


def test_free_boundary_probe_refines_without_full_ladder(monkeypatch):
    from vmex.core import multigrid, wout

    states = [object(), object()]
    outputs = [
        SimpleNamespace(b0=2.0, Aminor_p=0.5),
        SimpleNamespace(b0=2.1, Aminor_p=0.55),
    ]

    def fake_solve(*args, **kwargs):
        return SimpleNamespace(
            state=states.pop(0),
            fsqr=0.0,
            fsqz=0.0,
            fsql=0.0,
            iterations=1,
            converged=True,
            vacuum=None,
        )

    monkeypatch.setattr(multigrid, "solve_free_boundary_multigrid", fake_solve)
    monkeypatch.setattr(
        multigrid, "interpolate_state", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(wout, "wout_from_state", lambda **kwargs: outputs.pop(0))
    inp = VmecInput(
        mpol=2,
        ntor=0,
        ns_array=[17],
        lfreeb=True,
        mgrid_file="mgrid.nc",
        rbc=np.array([[3.0, 0.5]]),
        zbs=np.array([[0.0, 0.5]]),
    )
    probe = probe_input(inp, mgrid_path="mgrid.nc", device="cpu")
    assert (probe.coarse_ns, probe.fine_ns) == (9, 17)
    assert probe.b0 == 2.1
    assert probe.aminor == 0.55


def test_mgrid_scaling_distinguishes_per_ampere_and_raw_tables():
    data = MgridData(
        rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5,
        ir=2, jz=2, kp=2, nfp=1, nextcur=1,
        mgrid_mode="S", coil_groups=("coil",), raw_coil_cur=(10.0,),
        br=np.ones((1, 2, 2, 2)),
        bp=2.0 * np.ones((1, 2, 2, 2)),
        bz=3.0 * np.ones((1, 2, 2, 2)),
    )
    scaled = scale_mgrid(data, b_scale=3.0, r_scale=2.0)
    assert (scaled.rmin, scaled.rmax, scaled.zmin, scaled.zmax) == (
        2.0, 4.0, -1.0, 1.0
    )
    assert scaled.raw_coil_cur == (60.0,)
    np.testing.assert_allclose(scaled.br, data.br / 2.0)

    raw = scale_mgrid(
        dataclasses.replace(data, mgrid_mode="R"),
        b_scale=3.0,
        r_scale=2.0,
    )
    np.testing.assert_allclose(raw.br, 3.0 * data.br)


@pytest.fixture(scope="module")
def finite_beta_similarity():
    jax.config.update("jax_disable_jit", False)
    inp = VmecInput.from_file(DATA / "input.nfp2_QA_finite_beta")
    inp = dataclasses.replace(
        inp,
        ns_array=np.array([7, 11]),
        ftol_array=np.array([1e-9, 1e-10]),
        niter_array=np.array([2000, 3000]),
    )
    scaled_inp = scale_input(inp, b_scale=2.3, r_scale=1.7)

    def run(deck):
        result = solve_multigrid(deck, verbose=False)
        return wout_from_state(
            inp=deck,
            state=result.state,
            fsqr=float(result.fsqr),
            fsqz=float(result.fsqz),
            fsql=float(result.fsql),
            niter=int(result.iterations),
            converged=bool(result.converged),
        )

    return run(inp), run(scaled_inp)


def test_scale_wout_commutes_with_finite_beta_solve(finite_beta_similarity):
    original, solved_scaled_input = finite_beta_similarity
    scaled_wout = scale_wout(original, b_scale=2.3, r_scale=1.7)
    _assert_wout_similarity(solved_scaled_input, scaled_wout)


def _assert_wout_similarity(actual, expected):
    operational = {
        "mgrid_file", "niter", "itfsq", "fsql", "fsqr", "fsqz", "fsqt", "wdot",
    }
    for field in dataclasses.fields(expected):
        name = field.name
        if name in operational:
            continue
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        if expected_value is None or isinstance(expected_value, (str, tuple, bool)):
            assert actual_value == expected_value, name
        else:
            expected_array = np.asarray(expected_value)
            error = np.linalg.norm(np.asarray(actual_value) - expected_array)
            limit = (
                2e-8 * np.linalg.norm(expected_array)
                + 2e-9 * np.sqrt(expected_array.size)
            )
            assert error <= limit, (name, error, limit)


def _assert_free_boundary_similarity(
    inp, mgrid, mgrid_path, tmp_path, *, b_scale, r_scale,
):
    def run(deck, grid, path):
        result = solve_free_boundary_multigrid(deck, mgrid_path=path, verbose=False)
        return wout_from_state(
            inp=deck,
            state=result.state,
            fsqr=float(result.fsqr),
            fsqz=float(result.fsqz),
            fsql=float(result.fsql),
            niter=int(result.iterations),
            converged=bool(result.converged),
            vacuum_output=result.vacuum,
            nextcur=grid.nextcur,
            extcur=deck.extcur,
            mgrid_mode=grid.mgrid_mode,
            curlabel=grid.coil_groups,
        )

    original = run(inp, mgrid, mgrid_path)
    scaled_input = scale_input(inp, b_scale=b_scale, r_scale=r_scale)
    scaled_mgrid = scale_mgrid(
        mgrid, b_scale=b_scale, r_scale=r_scale,
    )
    scaled_mgrid_path = tmp_path / f"{Path(mgrid_path).stem}_scaled.nc"
    write_mgrid(scaled_mgrid_path, scaled_mgrid)
    actual = run(scaled_input, scaled_mgrid, scaled_mgrid_path)
    _assert_wout_similarity(
        actual,
        scale_wout(original, b_scale=b_scale, r_scale=r_scale),
    )


@pytest.mark.full
def test_scale_symmetric_free_boundary_commutes_through_nestor(tmp_path):
    mgrid_path = DATA / "mgrid_cth_like.nc"
    if not mgrid_path.exists():
        pytest.skip("mgrid_cth_like.nc not fetched (tools/fetch_assets.py)")
    _assert_free_boundary_similarity(
        VmecInput.from_file(DATA / "input.cth_like_free_bdy"),
        read_mgrid(mgrid_path),
        mgrid_path,
        tmp_path,
        b_scale=1.4,
        r_scale=1.2,
    )


@pytest.mark.full
def test_scale_lasym_free_boundary_commutes_through_nestor(tmp_path):
    from test_lasym_free_case import lasym_free_input, lasym_free_mgrid_data

    mgrid = lasym_free_mgrid_data()
    mgrid_path = write_mgrid(tmp_path / "mgrid_d3d_lasym.nc", mgrid)
    _assert_free_boundary_similarity(
        lasym_free_input(DATA),
        mgrid,
        mgrid_path,
        tmp_path,
        b_scale=1.25,
        r_scale=1.5,
    )


def test_aries_cs_wout_targets(finite_beta_similarity):
    original, _ = finite_beta_similarity
    inp = VmecInput.from_file(DATA / "input.nfp2_QA_finite_beta")
    assert input_minor_radius(inp) == pytest.approx(original.Aminor_p, rel=2e-14)
    b_scale, r_scale = aries_cs_scales(original)
    scaled = scale_wout(original, b_scale=b_scale, r_scale=r_scale)
    assert abs(scaled.b0) == pytest.approx(5.7)
    assert scaled.Aminor_p == pytest.approx(1.7)
    with pytest.raises(ValueError, match="nonzero b0"):
        aries_cs_scales(dataclasses.replace(original, b0=0.0))


def test_boozer_transform_obeys_same_similarity(finite_beta_similarity, tmp_path):
    pytest.importorskip("booz_xform_jax")
    netcdf4 = pytest.importorskip("netCDF4")
    from vmex.core.boozer import run_booz_xform

    original, scaled = finite_beta_similarity
    paths = []
    for name, wout in (("original", original), ("scaled", scaled)):
        wout_path = write_wout(tmp_path / f"wout_{name}.nc", wout)
        paths.append(run_booz_xform(
            wout_path,
            mbooz=12,
            nbooz=12,
            output_path=tmp_path / f"boozmn_{name}.nc",
        ))

    factors = {
        "aspect_b": 1.0,
        "toroidal_flux_b": 2.3 * 1.7**2,
        "iota_b": 1.0,
        "buco_b": 2.3 * 1.7,
        "bvco_b": 2.3 * 1.7,
        "bmnc_b": 2.3,
        "rmnc_b": 1.7,
        "zmns_b": 1.7,
        "numns_b": 1.0,
        "pmns_b": 1.0,
        "gmn_b": 1.7 / 2.3,
    }
    with netcdf4.Dataset(paths[0]) as first, netcdf4.Dataset(paths[1]) as second:
        for name, factor in factors.items():
            np.testing.assert_allclose(
                second.variables[name][...],
                factor * first.variables[name][...],
                rtol=2e-9,
                atol=2e-11,
            )


def test_cli_explicit_input_and_default_wout_scaling(
    finite_beta_similarity, tmp_path,
):
    input_path = DATA / "input.nfp2_QA_finite_beta"
    assert cli.main([
        "--scale", str(input_path), "2", "3", "--outdir", str(tmp_path), "--quiet",
    ]) == 0
    scaled_input = VmecInput.from_file(
        tmp_path / "input.nfp2_QA_finite_beta_scaled"
    )
    original_input = VmecInput.from_file(input_path)
    assert scaled_input.phiedge == 18.0 * original_input.phiedge

    original_wout, _ = finite_beta_similarity
    wout_path = write_wout(tmp_path / "wout_case.nc", original_wout)
    assert cli.main([
        "--scale", str(wout_path), "--outdir", str(tmp_path), "--quiet",
    ]) == 0
    scaled_wout = read_wout(tmp_path / "wout_case_scaled.nc")
    assert abs(scaled_wout.b0) == pytest.approx(5.7)
    assert scaled_wout.Aminor_p == pytest.approx(1.7)


def test_cli_scales_free_boundary_mgrid_sidecar(tmp_path):
    mgrid = MgridData(
        rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5,
        ir=2, jz=2, kp=2, nfp=1, nextcur=1,
        mgrid_mode="S", coil_groups=("coil",), raw_coil_cur=(10.0,),
        br=np.ones((1, 2, 2, 2)),
        bp=np.ones((1, 2, 2, 2)),
        bz=np.ones((1, 2, 2, 2)),
    )
    write_mgrid(tmp_path / "mgrid.nc", mgrid)
    input_path = VmecInput(
        mpol=2,
        ntor=0,
        lfreeb=True,
        mgrid_file="mgrid.nc",
        extcur=[10.0],
        rbc=np.array([[3.0, 0.5]]),
        zbs=np.array([[0.0, 0.5]]),
    ).to_indata(tmp_path / "input.free")
    assert cli.main([
        "--scale", str(input_path), "3", "2", "--outdir", str(tmp_path),
    ]) == 0
    scaled_input = VmecInput.from_file(tmp_path / "input.free_scaled")
    assert scaled_input.mgrid_file == "mgrid_scaled.nc"
    np.testing.assert_allclose(scaled_input.extcur, [60.0])
    scaled_mgrid = read_mgrid(tmp_path / scaled_input.mgrid_file)
    assert scaled_mgrid.rmax == 4.0
    np.testing.assert_allclose(scaled_mgrid.br, 0.5)


def test_cli_scaling_validation_and_json(tmp_path):
    inp = VmecInput(
        mpol=2,
        ntor=0,
        rbc=np.array([[3.0, 0.5]]),
        zbs=np.array([[0.0, 0.5]]),
    )
    json_path = inp.to_json(tmp_path / "input.json")
    assert cli.main([
        "--scale", str(json_path), "2", "3", "--outdir", str(tmp_path), "--quiet",
    ]) == 0
    assert (tmp_path / "input_scaled.json").exists()
    assert cli.main(["--scale", str(json_path), "2", "--quiet"]) == INPUT_ERROR_FLAG
    assert cli.main([
        "--scale", str(json_path), "0", "3", "--quiet",
    ]) == INPUT_ERROR_FLAG
    with pytest.raises(SystemExit):
        cli.main([str(json_path), "2", "3"])
    with pytest.raises(SystemExit):
        cli.main(["--scale", str(json_path), "2", "3", "--booz"])

    for name, mgrid_file in (
        ("direct", "DIRECT_COILS"),
        ("missing", "missing_mgrid.nc"),
    ):
        path = dataclasses.replace(
            inp, lfreeb=True, mgrid_file=mgrid_file
        ).to_indata(tmp_path / f"input.{name}")
        assert cli.main([
            "--scale", str(path), "2", "3", "--quiet",
        ]) == INPUT_ERROR_FLAG


def test_cli_default_input_scaling_reconverges_to_aries_cs(tmp_path, capsys):
    input_path = VmecInput(
        mpol=2,
        ntor=0,
        ns_array=[3],
        ftol_array=[1e-10],
        niter_array=[1000],
        phiedge=1.0,
        ai=np.array([0.4]),
        rbc=np.array([[3.0, 0.5]]),
        zbs=np.array([[0.0, 0.5]]),
    ).to_indata(tmp_path / "input.tiny")
    assert cli.main([
        "--scale", str(input_path), "--outdir", str(tmp_path),
        "--device", "cpu",
    ]) == 0
    output = capsys.readouterr().out
    match = re.search(r"b0=.*?\(change ([0-9.eE+-]+)\)", output)
    assert match is not None
    declared_error = float(match.group(1))
    scaled_input = VmecInput.from_file(tmp_path / "input.tiny_scaled")
    result = solve_multigrid(scaled_input, verbose=False, device="cpu")
    scaled_wout = wout_from_state(
        inp=scaled_input,
        state=result.state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=bool(result.converged),
    )
    assert abs(abs(scaled_wout.b0) / 5.7 - 1.0) <= 2.0 * declared_error
    assert scaled_wout.Aminor_p == pytest.approx(1.7, rel=2e-12)
