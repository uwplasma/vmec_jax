"""Unit tests for ``tools/first_divergence.py`` (privacy-safe comparator).

Pure-parsing/classification tests run without a VMEC2000 binary; the single
live smoke test follows the ``tests/test_vmec2000_live.py`` convention and is
gated behind ``--run-vmec2000`` (optionally ``--vmec2000-executable PATH``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"

_SPEC = importlib.util.spec_from_file_location(
    "first_divergence", ROOT / "tools" / "first_divergence.py")
assert _SPEC is not None and _SPEC.loader is not None
fd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fd)


# ---------------------------------------------------------------------------
# (a) VMEC2000 log/threed1 termination classification (synthetic text)
# ---------------------------------------------------------------------------

_TABLE = (
    "  ITER    FSQR      FSQZ      FSQL      RAX(v=0)    DELT       WMHD\n"
    "     1  1.00E-01  2.00E-02  3.00E-03  3.999E+00  9.00E-01  1.4142E+00\n"
    "    40  1.00E-11  2.00E-12  3.00E-13  3.998E+00  9.00E-01  1.4141E+00\n"
)


def test_vmec2000_normal_termination_is_converged():
    text = _TABLE + " EXECUTION TERMINATED NORMALLY\n"
    assert fd.classify_vmec2000_termination(text) == fd.TERM_CONVERGED


def test_vmec2000_niter_exhaustion_is_iteration_budget():
    """NITER exhaustion must NOT be classified as convergence."""
    text = _TABLE + " MORE ITERATIONS REQUIRED\n"
    assert fd.classify_vmec2000_termination(text) == fd.TERM_ITERATION_BUDGET


def test_vmec2000_niter_hint_outranks_normal_banner():
    """Measured build: exhausted runs print the eqsolve hint AND the normal
    banner (wout ier_flag stays 0) -- the hint must win."""
    text = _TABLE + (
        " Try increasing NITER or PRE_NITER if the preconditioner is on.\n"
        "\n EXECUTION TERMINATED NORMALLY\n"
    )
    assert fd.classify_vmec2000_termination(text) == fd.TERM_ITERATION_BUDGET


def test_vmec2000_jac75_abort():
    text = _TABLE + " MORE THAN 75 JACOBIAN ITERATIONS (DECREASE DELT)\n"
    assert fd.classify_vmec2000_termination(text) == fd.TERM_JACOBIAN_75


def test_vmec2000_nan_table_row_is_non_finite():
    text = _TABLE + (
        "    41       NaN       NaN       NaN  3.998E+00  9.00E-01       NaN\n"
    )
    assert fd.classify_vmec2000_termination(text) == fd.TERM_NON_FINITE


def test_vmec2000_overflow_asterisks_row_is_non_finite():
    text = _TABLE + (
        "    41  ********  1.00E+30  3.00E-13  3.998E+00  9.00E-01  1.41E+00\n"
    )
    assert fd.classify_vmec2000_termination(text) == fd.TERM_NON_FINITE


def test_vmec2000_ieee_trap_is_non_finite():
    text = _TABLE + "Fortran runtime warning: IEEE_INVALID_FLAG\n"
    assert fd.classify_vmec2000_termination(text) == fd.TERM_NON_FINITE


def test_vmec2000_input_rejection_markers():
    for marker in (
        "ERROR READING INPUT FILE OR NAMELIST",
        "ERROR IN INPUT VALUES",
        "PHIEDGE HAS WRONG SIGN IN VACUUM REGION",
        "NS ARRAY MUST NOT BE ALL ZEROES",
    ):
        assert fd.classify_vmec2000_termination(f" {marker}\n") == \
            fd.TERM_INPUT_ERROR


def test_vmec2000_unparseable_output_is_unknown():
    """No recognisable marker (e.g. crash before output) -> UNKNOWN."""
    assert fd.classify_vmec2000_termination("") == fd.TERM_UNKNOWN
    assert fd.classify_vmec2000_termination(
        "some unrelated shell noise\n") == fd.TERM_UNKNOWN
    # a clean iteration table with no termination banner at all
    assert fd.classify_vmec2000_termination(_TABLE) == fd.TERM_UNKNOWN


def test_vmec2000_word_boundaries_do_not_false_positive():
    """'information'/'nano' inside prose must not read as Inf/NaN."""
    text = "     1  information about nanoseconds elapsed\n"
    assert fd.classify_vmec2000_termination(text) == fd.TERM_UNKNOWN


# ---------------------------------------------------------------------------
# (b) VMEX termination classification via the typed exception taxonomy
#     (classifier called directly -- no solves)
# ---------------------------------------------------------------------------

def test_vmex_converged_flag_classification():
    assert fd.classify_vmex_termination(converged=True) == fd.TERM_CONVERGED
    assert fd.classify_vmex_termination(converged=False) == \
        fd.TERM_ITERATION_BUDGET


def test_vmex_typed_exceptions_map_to_classes():
    from vmex.core import errors

    cases = [
        (errors.VmecConvergenceError("m"), fd.TERM_ITERATION_BUDGET),
        (errors.VmecJacobianError("m"), fd.TERM_JACOBIAN_75),
        (errors.VmecNumericalError("m"), fd.TERM_NON_FINITE),
        (errors.VmecInputError("m"), fd.TERM_INPUT_ERROR),
        (errors.MgridNotFoundError("m"), fd.TERM_INPUT_ERROR),
    ]
    for exc, expected in cases:
        assert fd.classify_vmex_termination(exc=exc) == expected, type(exc)


def test_vmex_base_error_falls_back_to_ier_flag():
    from vmex.core import errors

    for flag, expected in [
        (errors.MORE_ITER_FLAG, fd.TERM_ITERATION_BUDGET),
        (errors.BAD_JACOBIAN_FLAG, fd.TERM_JACOBIAN_75),
        (errors.JAC75_FLAG, fd.TERM_JACOBIAN_75),
        (errors.INPUT_ERROR_FLAG, fd.TERM_INPUT_ERROR),
        (errors.PHIEDGE_ERROR_FLAG, fd.TERM_INPUT_ERROR),
        (errors.NS_ERROR_FLAG, fd.TERM_INPUT_ERROR),
        (errors.MISC_ERROR_FLAG, fd.TERM_UNKNOWN),
    ]:
        exc = errors.VmecError("m", ier_flag=flag)
        assert fd.classify_vmex_termination(exc=exc) == expected, flag


def test_vmex_untyped_exception_is_unknown():
    assert fd.classify_vmex_termination(exc=ValueError("boom")) == \
        fd.TERM_UNKNOWN


def test_acceptance_status_strings_are_class_only():
    assert fd._acceptance(fd.TERM_CONVERGED) == "ACCEPTED"
    assert fd._acceptance(fd.TERM_ITERATION_BUDGET) == "ACCEPTED"
    assert fd._acceptance(fd.TERM_INPUT_ERROR) == "REJECTED(INPUT_ERROR)"
    assert fd._acceptance(fd.TERM_NON_FINITE) == "ACCEPTED_THEN_NON_FINITE"
    assert fd._acceptance(fd.TERM_UNKNOWN, "TypeError") == \
        "ACCEPTED_THEN_UNKNOWN(TypeError)"


# ---------------------------------------------------------------------------
# (c) relative MGRID_FILE resolution against the deck directory
# ---------------------------------------------------------------------------

def test_relative_mgrid_resolves_against_deck_directory(tmp_path):
    deck_dir = tmp_path / "deck"
    (deck_dir / "fields").mkdir(parents=True)
    (deck_dir / "fields" / "mgrid_case.nc").write_bytes(b"mgrid-bytes")
    src = deck_dir / "input.case"
    src.write_text(
        "&INDATA\n"
        "  LFREEB = T,\n"
        "  MGRID_FILE = 'fields/mgrid_case.nc',\n"
        "  NSTEP = 10,\n"
        "  NITER_ARRAY = 500, 1000,\n"
        "/\n")
    work = tmp_path / "work"
    work.mkdir()
    dst = fd._prepare_deck(src, work, 25)
    assert dst.parent == work
    # copied next to the prepared deck ...
    assert (work / "mgrid_case.nc").read_bytes() == b"mgrid-bytes"
    # ... and re-referenced by basename so both codes resolve it from cwd
    m = re.search(r"MGRID_FILE\s*=\s*'([^']+)'", dst.read_text())
    assert m is not None and m.group(1) == "mgrid_case.nc"
    assert re.search(r"NSTEP\s*=\s*1\b", dst.read_text())


def test_missing_mgrid_leaves_deck_reference_unchanged(tmp_path):
    src = tmp_path / "input.nomgrid"
    src.write_text(
        "&INDATA\n  LFREEB = T,\n  MGRID_FILE = 'mgrid_absent.nc',\n"
        "  NSTEP = 3,\n/\n")
    work = tmp_path / "w"
    work.mkdir()
    dst = fd._prepare_deck(src, work, None)
    assert "mgrid_absent.nc" in dst.read_text()
    assert not (work / "mgrid_absent.nc").exists()


# ---------------------------------------------------------------------------
# (d) lowercase and multiline namelist robustness of the deck rewriter
# ---------------------------------------------------------------------------

def test_lowercase_and_multiline_namelist(tmp_path):
    deck_dir = tmp_path / "deck"
    deck_dir.mkdir()
    (deck_dir / "mgrid_low.nc").write_bytes(b"x")
    src = deck_dir / "input.low"
    src.write_text(
        "&indata\n"
        "  lfreeb = t\n"
        '  mgrid_file = "mgrid_low.nc"\n'
        "  nstep = 250\n"
        "  niter_array = 500 1000\n"
        "     2000\n"
        "  ftol_array = 1e-14\n"
        "/\n")
    work = tmp_path / "w"
    work.mkdir()
    dst = fd._prepare_deck(src, work, 40)
    text = dst.read_text()
    # lowercase nstep rewritten (key case preserved, value forced to 1)
    assert re.search(r"nstep\s*=\s*1\b", text, re.I)
    assert "250" not in text
    # the multiline niter_array continuation is replaced as one block
    assert "NITER_ARRAY = 40, 40, 40, 40, 40" in text
    assert "500" not in text and "2000" not in text
    # lowercase, double-quoted mgrid_file still found and copied
    assert (work / "mgrid_low.nc").exists()
    assert "ftol_array = 1e-14" in text  # neighbouring key untouched


def test_nstep_inserted_when_missing_lowercase_indata(tmp_path):
    src = tmp_path / "input.nonstep"
    src.write_text("&indata\n  mpol = 4\n/\n")
    work = tmp_path / "w2"
    work.mkdir()
    dst = fd._prepare_deck(src, work, None)
    assert re.search(r"NSTEP\s*=\s*1\b", dst.read_text(), re.I)


# ---------------------------------------------------------------------------
# (e) privacy: harness failures must never echo the deck path
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# (a2) threed1 input-echo parsing + C1 sub-stage comparison
# ---------------------------------------------------------------------------

_ECHO_HEAD = (
    " R-Z FOURIER BOUNDARY COEFFICIENTS AND MAGNETIC AXIS INITIAL GUESS\n"
    " --------------------------------------------------------------\n"
    "   nb  mb     rbc         rbs         zbc         zbs       raxis(c)\n"
)


def _echo_from_input(deck: Path, doctor=None) -> str:
    """A threed1-style boundary echo rendered from VMEX's own parse."""
    import numpy as np

    from vmex.core.input import VmecInput
    inp = VmecInput.from_file(str(deck))
    rbc, rbs = np.asarray(inp.rbc, float), np.asarray(inp.rbs, float)
    zbc, zbs = np.asarray(inp.zbc, float), np.asarray(inp.zbs, float)
    lines = [_ECHO_HEAD.rstrip("\n")]
    for mb in range(int(inp.mpol)):
        for nb in range(-int(inp.ntor), int(inp.ntor) + 1):
            row = [rbc[inp.ntor + nb, mb], rbs[inp.ntor + nb, mb],
                   zbc[inp.ntor + nb, mb], zbs[inp.ntor + nb, mb]]
            if doctor is not None:
                row = doctor(nb, mb, row)
            if not any(abs(v) > 0 for v in row):
                continue
            vals = "  ".join(f"{v: .4E}" for v in row)
            lines.append(f"   {nb:>2d}  {mb:>2d}  {vals}")
    return "\n".join(lines) + "\n\n NEXT SECTION\n"


def _run_echo_compare(ref_text: str, deck: Path) -> tuple[list[str], list[str]]:
    stages: list[str] = []
    klasses: list[str] = []

    def stage(code: str, text: str) -> None:
        stages.append(f"{code}: {text}")

    fd._compare_parsed_inputs(ref_text, deck, stage, klasses.append, False)
    return stages, klasses


def test_threed1_scalar_and_coeff_parsers():
    text = (
        "    nfp      gamma      spres_ped    phiedge(wb)     curtor(A)        lRFP\n"
        "      5  0.000E+00      1.000E+00     -3.500E-02     4.323E+04           F\n"
        "  ncurr  niter   nsin  nstep  nvacskip      ftol     tcon0    lasym  lforbal lmove_axis lconm1\n"
        "      1   2500     15    100         9  1.00E-10  1.00E+00        F        F        T        T\n"
        " Pressure profile factor:  4.3229E+02 (multiplier for pressure)\n"
        " MASS PROFILE COEFFICIENTS - newton/m**2 (EXPANSION IN NORMALIZED RADIUS):\n"
        " PMASS parameterization type is 'two_power'\n"
        " -----------------------------------\n"
        "   1.000E+00   5.000E+00   1.000E+01\n"
    )
    scal = fd._threed1_scalars(text)
    assert scal["nfp"] == 5 and scal["phiedge"] == pytest.approx(-0.035)
    assert scal["curtor"] == pytest.approx(4.323e4)
    assert scal["lforbal"] == 0.0 and scal["pres_scale"] == pytest.approx(432.29)
    assert fd._threed1_coeffs(text, "MASS PROFILE COEFFICIENTS") == [1.0, 5.0, 10.0]
    assert fd._threed1_coeffs(text, "IOTA PROFILE COEFFICIENTS") is None


def test_parse_echo_matches_own_render():
    deck = DATA / "input.cth_like_fixed_bdy"
    stages, klasses = _run_echo_compare(_echo_from_input(deck), deck)
    assert any("C1 PARSE_BOUNDARY: MATCH" in s for s in stages)
    assert all(k == fd.MATCH for k in klasses)


def test_parse_echo_flags_doctored_boundary_mode():
    deck = DATA / "input.cth_like_fixed_bdy"

    def doctor(nb, mb, row):
        return [2.0 * v for v in row] if (nb, mb) == (0, 1) else row

    stages, klasses = _run_echo_compare(
        _echo_from_input(deck, doctor=doctor), deck)
    line = next(s for s in stages if s.startswith("C1 PARSE_BOUNDARY"))
    assert "DIVERGENT" in line and "(nb=0,mb=1)" in line
    assert fd.DIVERGENT in klasses
    # privacy: no coefficient values in the default (no --details) output
    assert not re.search(r"\d\.\d{3,}E[+-]\d", line)


def test_parse_echo_flags_vmex_only_mode():
    """A mode VMEX parsed as nonzero but absent from the echo is a finding."""
    deck = DATA / "input.cth_like_fixed_bdy"

    def doctor(nb, mb, row):
        return [0.0] * 4 if (nb, mb) == (0, 2) else row

    stages, _ = _run_echo_compare(_echo_from_input(deck, doctor=doctor), deck)
    line = next(s for s in stages if s.startswith("C1 PARSE_BOUNDARY"))
    assert "DIVERGENT" in line and "(nb=0,mb=2)" in line


def test_privacy_usage_error_hides_deck_path(tmp_path, capsys, monkeypatch):
    marker = "CONFIDENTIAL_MARKER_7QX"
    deck = tmp_path / marker / "input.private"  # directory does not exist
    monkeypatch.setattr(sys, "argv", [
        "first_divergence.py", str(deck),
        "--xvmec2000", str(tmp_path / "no_such_exe")])
    rc = fd.main()
    captured = capsys.readouterr()
    assert rc == 3
    assert marker not in captured.out + captured.err
    assert "C0 USAGE_ERROR input file not found" in captured.out


def test_usage_input_directory_is_path_free_hint(tmp_path, capsys, monkeypatch):
    marker = "CONFIDENTIAL_MARKER_4TD"
    deck_dir = tmp_path / marker
    deck_dir.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "first_divergence.py", str(deck_dir),
        "--xvmec2000", str(tmp_path / "xvmec2000")])
    rc = fd.main()
    captured = capsys.readouterr()
    assert rc == 3
    assert marker not in captured.out + captured.err
    assert "C0 USAGE_ERROR input is a directory" in captured.out


def test_usage_executable_directory_without_binary(tmp_path, capsys, monkeypatch):
    deck = tmp_path / "input.case"
    deck.write_text("&INDATA\n  NSTEP = 5,\n/\n")
    exe_dir = tmp_path / "build"
    exe_dir.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "first_divergence.py", str(deck), "--xvmec2000", str(exe_dir)])
    rc = fd.main()
    captured = capsys.readouterr()
    assert rc == 3
    assert "C0 USAGE_ERROR --xvmec2000 is a directory" in captured.out


def test_usage_executable_directory_resolves_to_binary(tmp_path, capsys,
                                                       monkeypatch):
    deck = tmp_path / "input.case"
    deck.write_text("&INDATA\n  NSTEP = 5,\n/\n")
    exe_dir = tmp_path / "build"
    exe_dir.mkdir()
    (exe_dir / "xvmec2000").write_text("")  # not runnable: C1 must follow
    monkeypatch.setattr(sys, "argv", [
        "first_divergence.py", str(deck), "--xvmec2000", str(exe_dir)])
    rc = fd.main()
    captured = capsys.readouterr()
    assert "USAGE_ERROR" not in captured.out
    assert "using the xvmec2000 executable found inside" in captured.out
    assert rc == 2  # proceeds into compare; the stub cannot actually run
    assert "C1 PARSE: VMEC2000 run failed" in captured.out


def test_privacy_missing_executable_hides_paths(tmp_path, capsys, monkeypatch):
    marker = "CONFIDENTIAL_MARKER_9ZK"
    deck_dir = tmp_path / marker
    deck_dir.mkdir()
    deck = deck_dir / "input.case"
    deck.write_text("&INDATA\n  NSTEP = 5,\n/\n")
    monkeypatch.setattr(sys, "argv", [
        "first_divergence.py", str(deck),
        "--xvmec2000", str(deck_dir / "missing_exe")])
    rc = fd.main()
    captured = capsys.readouterr()
    assert rc == 3
    assert marker not in captured.out + captured.err
    assert "C0 USAGE_ERROR --xvmec2000 executable not found" in captured.out


def test_privacy_broken_executable_hides_paths(tmp_path, capsys, monkeypatch):
    """A present-but-unrunnable executable exercises compare's internal C1
    error path; the report must stay path-free."""
    marker = "CONFIDENTIAL_MARKER_9ZK"
    deck_dir = tmp_path / marker
    deck_dir.mkdir()
    deck = deck_dir / "input.case"
    deck.write_text("&INDATA\n  NSTEP = 5,\n/\n")
    broken = deck_dir / "broken_exe"
    broken.write_text("")  # exists but is not runnable
    monkeypatch.setattr(sys, "argv", [
        "first_divergence.py", str(deck), "--xvmec2000", str(broken)])
    rc = fd.main()
    captured = capsys.readouterr()
    assert rc == 2
    assert marker not in captured.out + captured.err
    assert "C1 PARSE: VMEC2000 run failed" in captured.out


# ---------------------------------------------------------------------------
# live smoke (needs xvmec2000; --run-vmec2000 convention)
# ---------------------------------------------------------------------------

def _executable(pytestconfig) -> Path:
    configured = str(pytestconfig.getoption("--vmec2000-executable")).strip()
    candidates = [Path(configured) if configured else None]
    discovered = shutil.which("xvmec2000")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    pytest.fail(
        "--run-vmec2000 requested but xvmec2000 was not found; pass "
        "--vmec2000-executable PATH"
    )


@pytest.mark.vmec2000_live
def test_live_smoke_solovev_reports_all_stages(pytestconfig, tmp_path):
    """The comparator runs both codes on the public solovev deck end-to-end."""
    exe = _executable(pytestconfig)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "first_divergence.py"),
         str(DATA / "input.solovev"), "--xvmec2000", str(exe),
         "--niter", "40"],
        capture_output=True, text=True, timeout=900, cwd=tmp_path,
        env={**os.environ, "JAX_ENABLE_X64": "1"})
    out = proc.stdout
    assert proc.returncode in (0, 1), (proc.returncode, out, proc.stderr)
    for code in ("C1 PARSE", "C2 AXIS_ROW1", "C3 ITER1_FORCES",
                 "C4 TRAJECTORY", "C7 RECOVERY", "C8 TERMINATION"):
        assert code in out, out
    # capped at 40 iterations, neither code can reach FTOL=1e-14: the C8
    # classes must agree on ITERATION_BUDGET rather than claim convergence
    assert "C8 TERMINATION: MATCH (ref=ITERATION_BUDGET "\
           "vmex=ITERATION_BUDGET)" in out, out
    assert "assessment: FIRST_DIVERGENCE_" in out
