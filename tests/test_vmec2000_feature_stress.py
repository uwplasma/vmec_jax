"""Public integration stress cases for VMEC2000 compatibility controls.

The high-mode case is constructed only from
``examples/data/input.serial2500170_surface_points_mpol12_ntor12``.  Its
boundary is already public and tracked in this repository.  The transformed
header deliberately combines the compatibility seams that are easy to miss
when tested separately: Fortran ordered overlays and array sections, a
four-stage radial ladder, APHI starting-element assignment, no supplied axis,
``LFORBAL=T``, and ``PRECON_TYPE='NONE'`` (VMEC's ordinary 1-D path).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tools.diagnose_input import diagnose
from vmex.core.errors import MORE_ITER_FLAG
from vmex.core.fourier import mode_table
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid


REPO = Path(__file__).resolve().parents[1]
PUBLIC_BOUNDARY = (
    REPO / "examples" / "data"
    / "input.serial2500170_surface_points_mpol12_ntor12"
)


def _replace_assignment(text: str, name: str, replacement: str) -> str:
    result, count = re.subn(
        rf"(?im)^\s*{re.escape(name)}\s*=.*$",
        replacement,
        text,
        count=1,
    )
    assert count == 1, name
    return result


def public_feature_stress_text() -> str:
    """Return the reproducible 238-mode public VMEC2000 stress deck."""
    text = PUBLIC_BOUNDARY.read_text()
    text = _replace_assignment(text, "MPOL", "  MPOL = 13")
    text = _replace_assignment(text, "NTOR", "  NTOR = 9")
    text = _replace_assignment(
        text,
        "ns_array",
        "  NS_ARRAY = 21, 34, 55, 89\n"
        "  NS_ARRAY(1) = 21, 34",
    )
    text = _replace_assignment(
        text,
        "niter_array",
        "  NITER_ARRAY = 1000, 1000, 4000, 10000",
    )
    text = _replace_assignment(
        text,
        "ftol_array",
        "  FTOL_ARRAY = 1e-6, 1e-11, 1.5e-9, 5e-14\n"
        "  FTOL_ARRAY = 1e-7, 1e-7",
    )
    text = _replace_assignment(text, "DELT", "  DELT = 0.5")
    text = text.replace(
        "  NCURR   = 1",
        "  APHI(1) = 1.0, 0.0\n"
        "  PRECON_TYPE = 'NONE'\n"
        "  LFORBAL = T\n"
        "  NCURR   = 1",
        1,
    )
    # Exercise a compact multidimensional section and later scalar overlays
    # without changing the public boundary: the existing RBC(n,0) statements
    # below overwrite the relevant values in normal Fortran source order.
    text = text.replace(
        "  ! VMEC coefficient order is (n,m): RBC(n,m), ZBS(n,m).",
        "  ! VMEC coefficient order is (n,m): RBC(n,m), ZBS(n,m).\n"
        "  RBC(-6:6,0) = 13*0.0\n"
        "  ZBS(-6:6,0) = 13*0.0",
        1,
    )
    return text


def test_public_stress_parser_and_first_force_diagnostic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every combined compatibility seam reaches a finite first force."""
    text = public_feature_stress_text()
    inp = VmecInput.from_indata_text(text)

    assert mode_table(inp.mpol, inp.ntor).mnmax == 238
    np.testing.assert_array_equal(inp.ns_array, [21, 34, 55, 89])
    np.testing.assert_array_equal(
        inp.niter_array, [1000, 1000, 4000, 10000]
    )
    np.testing.assert_array_equal(
        inp.ftol_array, [1e-7, 1e-7, 1.5e-9, 5e-14]
    )
    np.testing.assert_array_equal(inp.aphi[:2], [1.0, 0.0])
    assert inp.lforbal
    assert inp.precon_type.upper() == "NONE"
    assert not np.any(inp.raxis_c)
    assert not np.any(inp.zaxis_s)

    path = tmp_path / "input.public_feature_stress"
    path.write_text(text)
    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "input parsing: PASS" in output
    assert "input physics mode supported: PASS" in output
    assert "automatic first-pass axis recovery: REQUIRED" in output
    assert "radial=PASS" in output
    assert "assessment: OK_FIRST_FORCE_PASS_FINITE" in output


@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_public_238_mode_lforbal_solve_matches_vmec2000() -> None:
    """The difficult fixed-boundary proxy converges to the VMEC2000 result."""
    inp = VmecInput.from_indata_text(public_feature_stress_text())
    lines: list[str] = []
    result = solve_multigrid(
        inp,
        ns_array=[21],
        ftol_array=[1e-7],
        niter_array=[1000],
        verbose=True,
        emit=lambda value="", end="\n": lines.append(str(value) + end),
        device="cpu",
    )
    output = "".join(lines)

    assert result.converged
    assert result.iterations == 462
    assert result.jacobian_resets == 7
    assert output.count("TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS") == 1
    np.testing.assert_allclose(
        [result.fsqr, result.fsqz, result.fsql],
        [9.78117163e-8, 3.13954504e-8, 7.69528620e-9],
        rtol=2e-8,
    )
    np.testing.assert_allclose(
        [result.wb, result.r00, result.wmhd],
        [0.033530284026509774, 1.235051198548, 1.323722555191],
        rtol=2e-11,
        atol=2e-13,
    )


@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_public_238_mode_full_ladder_matches_vmec2000_trajectory() -> None:
    """All four radial stages reproduce the public VMEC2000 trajectory."""
    inp = VmecInput.from_indata_text(public_feature_stress_text())
    lines: list[str] = []
    result = solve_multigrid(
        inp,
        verbose=True,
        emit=lambda value="", end="\n": lines.append(str(value) + end),
        device="cpu",
        raise_on_max_iterations=False,
    )
    output = "".join(lines)

    assert not result.converged
    assert result.ier_flag == MORE_ITER_FLAG
    assert result.iterations == 10000
    assert result.jacobian_resets == 2
    for stage in (
        "NS =   21 NO. FOURIER MODES =  238 FTOLV =  1.000E-07 NITER =   1000",
        "NS =   34 NO. FOURIER MODES =  238 FTOLV =  1.000E-07 NITER =   1000",
        "NS =   55 NO. FOURIER MODES =  238 FTOLV =  1.500E-09 NITER =   4000",
        "NS =   89 NO. FOURIER MODES =  238 FTOLV =  5.000E-14 NITER =  10000",
    ):
        assert stage in output

    # These are the first/final screen rows produced by the same public deck
    # with the local VMEC2000 Release executable.  In particular, the first
    # fine-grid rows pin initialize_radial.f's residual-state continuation.
    for row in (
        "  462  9.78E-08  3.14E-08  7.70E-09",
        "    1  2.70E-03  1.31E-03  7.93E-06",
        "  139  9.84E-08  9.84E-08  6.19E-09",
        "    1  2.41E-03  1.12E-03  2.24E-06",
        "  543  1.14E-09  1.48E-09  1.48E-10",
        "    1  1.51E-03  8.12E-04  1.10E-06",
        "10000  3.06E-11  6.00E-11  3.11E-12",
    ):
        assert row in output
