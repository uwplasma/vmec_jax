"""Focused CLI device-selection parsing and solver forwarding tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vmex.core import cli, multigrid


def test_device_option_parses_supported_choices():
    parser = cli.build_parser()
    assert parser.parse_args(["input.case"]).device == "auto"
    assert parser.parse_args(["input.case", "--device", "none"]).device == "none"
    assert parser.parse_args(["input.case", "--device", "gpu"]).device == "gpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["input.case", "--device", "quantum"])


@pytest.mark.parametrize(("choice", "expected"), [("none", None), ("cpu", "cpu")])
def test_fixed_boundary_cli_forwards_device(monkeypatch, tmp_path, choice, expected):
    args = cli.build_parser().parse_args(["input.case", "--quiet", "--device", choice])
    inp = object()
    seen = {}

    monkeypatch.setattr(cli, "_read_input", lambda _: inp)
    monkeypatch.setattr(cli, "_free_boundary_plan", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_stage_overrides", lambda *a, **k: (None, None))
    monkeypatch.setattr(cli, "_write_wout_from_result", lambda *a, **k: object())

    def fake_solve(source, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(converged=True, ier_flag=0)

    monkeypatch.setattr(multigrid, "solve_multigrid", fake_solve)
    assert cli._solve_input_file(args, tmp_path / "input.case", tmp_path, emit=print) == 0
    assert seen["device"] == expected
    assert seen["release_stage_cache"] is True


def test_free_boundary_cli_forwards_device(monkeypatch, tmp_path):
    args = cli.build_parser().parse_args(
        ["input.case", "--quiet", "--device", "gpu"]
    )
    inp = SimpleNamespace(ns_array=[11], lfull3d1out=False)
    plan = SimpleNamespace(solver_kwargs={})
    seen = {}
    ftol_array, niter_array = [1e-8], [3]

    monkeypatch.setattr(cli, "_read_input", lambda _: inp)
    monkeypatch.setattr(cli, "_free_boundary_plan", lambda *a, **k: plan)
    monkeypatch.setattr(
        cli, "_stage_overrides", lambda *a, **k: (ftol_array, niter_array),
    )
    monkeypatch.setattr(cli, "_write_wout_from_result", lambda *a, **k: object())

    def fake_solve(source, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(converged=True, ier_flag=0)

    monkeypatch.setattr(multigrid, "solve_free_boundary_multigrid", fake_solve)
    assert cli._solve_input_file(args, tmp_path / "input.case", tmp_path, emit=print) == 0
    assert seen["device"] == "gpu"
    assert seen["ftol_array"] is ftol_array
    assert seen["niter_array"] is niter_array
    assert seen["raise_on_max_iterations"] is True
