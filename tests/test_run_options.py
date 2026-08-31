"""VMEX execution directives: parsing, precedence, and solve_file semantics.

The contract under test (plan section 8): run controls travel as VMEC-safe
comments or a reserved ``_vmex`` JSON section, ``VmecInput`` stays physics
only, precedence is ``CLI > Python keyword > file directive > default``, and
a failed polish behaves per ``polish_fail`` without a second solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vmex.core.errors import VmecInputError
from vmex.core.input import VmecInput
from vmex.core.run_options import (
    RunOptions,
    format_indata_directives,
    parse_indata_run_options,
    polish_config_from_options,
    read_input_request,
    resolve_run_options,
    strip_vmex_json,
)

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


# ---------------------------------------------------------------------------
# INDATA directive parsing
# ---------------------------------------------------------------------------


def test_all_directives_parse_together_with_inline_comments():
    options = parse_indata_run_options(
        "!@VMEX POLISH = AUTO\n"
        "  ! @ VMEX  POLISH_TOL = 2.5E-4   ! tighter than default\n"
        "!@vmex polish_fail = FALLBACK\n"
        "!@VMEX POLISH_DEGREE = 5\n"
        "&INDATA\nMPOL = 3\n/\n"
    )
    assert options == RunOptions(
        polish="auto", polish_tol=2.5e-4, polish_fail="fallback",
        polish_degree=5)


def test_no_directives_means_package_defaults():
    assert parse_indata_run_options("&INDATA\nMPOL = 3\n/\n") == RunOptions()


def test_legacy_polish_force_balance_spelling_still_parses():
    text = "! VMEX: POLISH_FORCE_BALANCE = .TRUE.\n&INDATA\n/\n"
    assert parse_indata_run_options(text).polish is True


def test_consistent_repetition_is_allowed_and_conflict_is_an_error():
    twice = "!@VMEX POLISH = AUTO\n!@VMEX POLISH = AUTO\n&INDATA\n/\n"
    assert parse_indata_run_options(twice).polish == "auto"
    conflict = "!@VMEX POLISH = AUTO\n!@VMEX POLISH = .FALSE.\n&INDATA\n/\n"
    with pytest.raises(VmecInputError, match="conflicting"):
        parse_indata_run_options(conflict)
    # The two spellings must agree with each other too.
    mixed = ("!@VMEX POLISH = .FALSE.\n"
             "! VMEX: POLISH_FORCE_BALANCE = .TRUE.\n&INDATA\n/\n")
    with pytest.raises(VmecInputError, match="conflicting"):
        parse_indata_run_options(mixed)


@pytest.mark.parametrize(
    "line, message",
    [
        ("!@VMEX POLISH = sometimes", "AUTO, TRUE, or FALSE"),
        ("!@VMEX POLISH_TOL = tight", "real number"),
        ("!@VMEX POLISH_TOL = -1.0", "positive"),
        ("!@VMEX POLISH_DEGREE = 4", "3, 5, 7"),
        ("!@VMEX POLISH_DEGREE = five", "integer"),
        ("!@VMEX POLISH_FAIL = explode", "error"),
        ("!@VMEX POLISH_MODE = auto", "unknown VMEX directive"),
    ],
)
def test_invalid_directives_fail_with_named_errors(line, message):
    with pytest.raises(VmecInputError, match=message):
        parse_indata_run_options(line + "\n&INDATA\n/\n")


def test_quoted_exclamation_marks_do_not_become_directives():
    """A '!' inside a namelist string is data; directives are comment lines."""
    text = "&INDATA\nMGRID_FILE = 'weird!@VMEX POLISH = AUTO'\nMPOL = 3\n/\n"
    assert parse_indata_run_options(text) == RunOptions()
    inp = VmecInput.from_indata_text(text)
    assert "!@VMEX" in inp.mgrid_file


def test_directive_round_trip_through_format():
    options = RunOptions(polish="auto", polish_tol=1e-8, polish_fail="warn",
                         polish_degree=7)
    text = format_indata_directives(options) + "&INDATA\nMPOL = 3\n/\n"
    assert parse_indata_run_options(text) == options
    assert format_indata_directives(RunOptions()) == ""


# ---------------------------------------------------------------------------
# JSON _vmex section
# ---------------------------------------------------------------------------


def test_json_vmex_section_round_trip(tmp_path):
    payload = {"mpol": 4, "ntor": 0,
               "_vmex": {"polish": "auto", "polish_degree": 5}}
    physics, options = strip_vmex_json(payload)
    assert "_vmex" not in physics and physics["mpol"] == 4
    assert options == RunOptions(polish="auto", polish_degree=5)

    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    request = read_input_request(path)
    assert request.options.polish == "auto"
    assert request.input.mpol == 4
    # VmecInput.from_file ignores execution metadata but keeps all physics.
    assert VmecInput.from_file(path).mpol == 4


@pytest.mark.parametrize(
    "section, message",
    [({"polish_mode": "auto"}, "unknown _vmex keys"),
     ("auto", "JSON object"),
     ({"polish": "sometimes"}, "AUTO, TRUE, or FALSE")],
)
def test_json_vmex_section_rejects_bad_content(section, message):
    with pytest.raises(VmecInputError, match=message):
        strip_vmex_json({"mpol": 4, "_vmex": section})


# ---------------------------------------------------------------------------
# Precedence and config mapping
# ---------------------------------------------------------------------------


def test_precedence_python_over_file_over_default():
    file_options = RunOptions(polish="auto", polish_degree=5)
    resolved, sources = resolve_run_options(file_options)
    assert resolved == file_options
    assert sources["polish"] == "file" and sources["polish_tol"] == "default"

    resolved, sources = resolve_run_options(file_options, polish=False,
                                            polish_tol=1e-6)
    assert resolved.polish is False and resolved.polish_tol == 1e-6
    assert resolved.polish_degree == 5          # untouched file value
    assert sources["polish"] == "python" and sources["polish_degree"] == "file"


def test_polish_config_mapping_and_explicit_config_priority():
    from vmex.core.polish_driver import PolishConfig

    options = RunOptions(polish=True, polish_tol=1e-5, polish_degree=5,
                         polish_fail="fallback")
    config = polish_config_from_options(options)
    assert config.tolerance == 1e-5
    assert config.radial_degree == 5
    assert config.fail_policy == "return_unpolished"
    # Nothing beyond driver defaults requested -> no config object at all,
    # keeping the plain path identical to before this module existed.
    assert polish_config_from_options(RunOptions(polish=True)) is None
    # An explicit PolishConfig wins over every directive-level scalar.
    base = PolishConfig(tolerance=3e-3)
    assert polish_config_from_options(options, base) is base


def test_plain_run_options_do_not_import_polish_driver(monkeypatch):
    """The default CLI path must not load the optional polishing stack."""
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.endswith("polish_driver"):
            raise AssertionError("plain run imported the polishing driver")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert polish_config_from_options(RunOptions()) is None


def test_public_python_polish_defaults_are_explicitly_off():
    import inspect

    import vmex as vj
    from vmex import optimize as opt

    for function in (vj.solve, vj.solve_multigrid, opt.solve_equilibrium):
        parameter = inspect.signature(function).parameters[
            "polish_force_balance"
        ]
        assert parameter.default is False
    # This file-oriented API alone needs a tri-state default: None means
    # honor an explicit VMEX directive in the input deck.
    assert inspect.signature(vj.solve_file).parameters["polish"].default is None


# ---------------------------------------------------------------------------
# solve_file
# ---------------------------------------------------------------------------


def test_solve_file_runs_and_writes_wout(tmp_path):
    import vmex as vj

    source = tmp_path / "input.solovev"
    source.write_text((DATA / "input.solovev").read_text(encoding="utf-8"),
                      encoding="utf-8")
    result = vj.solve_file(source, polish=False, outdir=tmp_path)
    assert result.converged
    wout_path = tmp_path / "wout_solovev.nc"
    assert wout_path.exists()
    from vmex.core.wout import read_wout

    assert int(read_wout(wout_path).ier_flag) == 0


def test_unpolished_wout_stays_bit_identical_to_the_plain_state_write(tmp_path):
    """The polished-export lane must not touch unpolished runs at all."""
    import dataclasses

    import vmex as vj
    from vmex.core.wout import wout_from_state, write_wout

    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(
        mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]))
    source = inp.to_indata(tmp_path / "input.unpolished_case")
    result = vj.solve_file(source, polish=False, outdir=tmp_path)
    assert result.polished_state is None
    assert result.native_equilibrium is None
    written = tmp_path / "wout_unpolished_case.nc"
    assert written.exists()

    # Reference: the direct state write with the CLI's own metadata,
    # bypassing every polish-aware branch in the writer.
    history = np.asarray(result.fsq_history, dtype=float)
    total = history[:, 0] + history[:, 1]
    stride = total.size // 100 + 1
    fsqt = total[stride - 1 :: stride][:100]
    reference = tmp_path / "wout_reference.nc"
    reparsed = read_input_request(source).input  # what solve_file solved
    write_wout(reference, wout_from_state(
        inp=reparsed, state=result.state, fsqr=float(result.fsqr),
        fsqz=float(result.fsqz), fsql=float(result.fsql), fsqt=fsqt,
        niter=int(result.iterations), converged=bool(result.converged),
        input_extension="unpolished_case", vacuum_output=result.vacuum))
    assert written.read_bytes() == reference.read_bytes()


def test_solve_file_rejects_polish_on_a_free_boundary_deck(tmp_path):
    source = tmp_path / "input.freeb"
    # MGRID_FILE must name something: readin.f (and VmecInput) force
    # LFREEB = F when it is 'NONE', which would silently reroute this deck to
    # the fixed-boundary branch and never reach the gate under test.
    source.write_text(
        "!@VMEX POLISH = .TRUE.\n"
        "&INDATA\nLFREEB = T\nMGRID_FILE = 'mgrid_missing.nc'\nMPOL = 3\n"
        "NS_ARRAY = 5\nRBC(0,0) = 1.0\nRBC(0,1) = 0.3\nZBS(0,1) = 0.3\n/\n",
        encoding="utf-8")
    import vmex as vj

    with pytest.raises(ValueError, match="fixed-boundary.*file"):
        vj.solve_file(source, write_wout=False)
    # --no-polish equivalent: the Python keyword overrides the directive, so
    # the gate no longer fires.  The deck then proceeds into the free-boundary
    # ladder and fails there for its own reason (no usable mgrid), which is
    # the point: the rejection above is the polish gate, not deck validation.
    with pytest.raises(Exception) as info:
        vj.solve_file(source, polish=False, write_wout=False,
                      niter_array=np.array([3]))
    assert "polishing requires" not in str(info.value)


def test_solve_file_polish_directive_activates_polishing(tmp_path):
    """One documented input runs the whole flow: directive -> polished wout.

    The deck is the fast certified case from the polish suite (solovev at
    mpol = 3, ns = 5 with the loose validation tolerance); the property under
    test is that the *file directive alone* turns polishing on, so the
    explicit PolishConfig only makes the certificate affordable, while the
    activation comes from the ``!@VMEX`` line.
    """
    import dataclasses

    import vmex as vj

    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(
        mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]))
    physics = inp.to_indata(tmp_path / "input.polished_case")
    source = tmp_path / "input.polished_case_directive"
    source.write_text("!@VMEX POLISH = .TRUE.\n"
                      + physics.read_text(encoding="utf-8"), encoding="utf-8")

    from vmex.core.polish_driver import PolishConfig

    config = PolishConfig(radial_degree=3, validation_tolerance=3.0)
    result = vj.solve_file(source, outdir=tmp_path, polish_config=config)
    assert result.converged
    assert result.polish_report is not None
    assert bool(result.polish_report.converged)
    assert result.polished_state is not None
    assert (tmp_path / "wout_polished_case_directive.nc").exists()

    # Without the directive the same solve does not polish: the activation
    # really came from the file.
    plain = vj.solve_file(physics, outdir=tmp_path, polish_config=config)
    assert plain.polish_report is None
