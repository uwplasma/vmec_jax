"""Regression tests for first-iteration NaN/Inf handling."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from vmex.core.errors import VmecNumericalError
from vmex.core.input import VmecInput
from vmex.core.solver import solve
from tools.diagnose_input import diagnose


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.usefixtures("_module_jit_enabled")
def test_zero_effective_toroidal_flux_fails_on_first_iteration() -> None:
    """Zero PHIEDGE used to print NaNs until NITER was exhausted."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    inp = dataclasses.replace(inp, phiedge=0.0, niter_array=[27])
    with pytest.raises(VmecNumericalError, match="NON-FINITE") as caught:
        solve(inp, max_iterations=27)
    assert caught.value.iteration == 1
    assert "PHIEDGE" in caught.value.hint


def test_shareable_diagnostic_redacts_input_details(capsys) -> None:
    """Default output must be safe to share for a confidential input deck."""
    path = DATA / "input.nfp2_QI"
    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "input.nfp2_QI" not in output
    assert "phiedge=" not in output.lower()
    assert "nfp=" not in output.lower()
    assert "R(0)=" not in output
    assert "FSQR=" not in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output


def test_shareable_diagnostic_warns_on_undersampled_angular_grid(
    tmp_path: Path, capsys
) -> None:
    """The aliasing warning discloses no resolution or coefficient values."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    path = dataclasses.replace(inp, ntheta=16, nzeta=14).to_indata(
        tmp_path / "input.private_undersampled"
    )

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "angular grid meets VMEC automatic-resolution floor: FAIL" in output
    assert "assessment: W01_ANGULAR_GRID_BELOW_VMEC_DEFAULT" in output
    assert "private_undersampled" not in output
    assert "ntheta=" not in output.lower()
    assert "nzeta=" not in output.lower()


def test_shareable_diagnostic_localizes_zero_energy_scale(
    tmp_path: Path, capsys
) -> None:
    """The public code identifies normalization failure without leaking values."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    private_path = dataclasses.replace(inp, phiedge=0.0).to_indata(
        tmp_path / "input.confidential_name"
    )

    assert diagnose(private_path) == 1
    output = capsys.readouterr().out
    assert "D03C_ZERO_ENERGY_SCALE" in output
    assert "unnormalized force sums finite: PASS" in output
    assert "input.confidential_name" not in output
    assert "phiedge=" not in output.lower()
    assert "FSQR=" not in output


def test_shareable_diagnostic_flags_unsupported_reconstruction(
    tmp_path: Path, capsys
) -> None:
    """VMEC reconstruction inputs fail explicitly instead of being ignored."""
    path = tmp_path / "input.private_reconstruction"
    path.write_text(
        """&INDATA
        LRECON = T
        IMSE = 1
        MPOL = 3
        NTOR = 0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )

    assert diagnose(path) == 1
    output = capsys.readouterr().out
    assert "D00A_RECONSTRUCTION_MODE_UNSUPPORTED" in output
    assert "private_reconstruction" not in output
    assert "RBC" not in output


def test_shareable_diagnostic_uses_production_mode_classification(
    tmp_path: Path, capsys,
) -> None:
    """A production-rejected model gets a stable value-free diagnostic code."""
    path = tmp_path / "input.private_trip3d"
    path.write_text("&INDATA\nTRIP3D_FILE='private_field_name.nc'\n/\n")
    assert diagnose(path) == 1
    output = capsys.readouterr().out
    assert "input parsing: PASS" in output
    assert "D00E_TRIP3D_MODE_UNSUPPORTED" in output
    assert "private_trip3d" not in output
    assert "private_field_name" not in output


def test_shareable_diagnostic_supports_lforbal(
    tmp_path: Path, capsys
) -> None:
    """The active non-variational force mode reaches the finite force audit."""
    path = tmp_path / "input.private_lforbal"
    path.write_text(
        """&INDATA
        LFORBAL = T
        MPOL = 3
        NTOR = 0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "input physics mode supported: PASS" in output
    assert "assessment: OK_FIRST_FORCE_PASS_FINITE" in output
    assert "D00D_LFORBAL_MODE_UNSUPPORTED" not in output
    assert "private_lforbal" not in output
    assert "RBC" not in output


def test_shareable_diagnostic_reports_high_force_axis_recovery(
    tmp_path: Path, capsys
) -> None:
    """A finite irst=4 trigger is reported without printing force values."""
    inp = VmecInput.from_file(DATA / "input.solovev")
    path = dataclasses.replace(
        inp, raxis_c=np.asarray([4.4]), lmove_axis=True,
    ).to_indata(tmp_path / "input.private_axis_recovery")

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "automatic first-pass axis recovery: REQUIRED" in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output
    assert "private_axis_recovery" not in output
    assert "4.4" not in output


def test_shareable_diagnostic_accepts_open_boundary_section(
    tmp_path: Path, capsys
) -> None:
    """Declared Fortran bounds make ``RBC(:,m)`` a legal array section."""
    path = tmp_path / "input.private_open_section"
    path.write_text(
        """&INDATA
        MPOL = 3
        NTOR = 1
        RBC(:,0) = 101*0.0, 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "input parsing: PASS" in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output
    assert "private_open_section" not in output
    assert "RBC" not in output


def test_inert_reconstruction_flag_matches_vmec2000_disable(
    tmp_path: Path, capsys
) -> None:
    """read_indata.f disables LRECON when no MSE or Thomson data are active."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    path = inp.to_indata(tmp_path / "input.inert_reconstruction")
    text = path.read_text().replace("&INDATA", "&INDATA\n  LRECON = T")
    path.write_text(text)

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "input physics mode supported: PASS" in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output


def test_indexed_aphi_prevents_false_zero_flux_diagnosis(
    tmp_path: Path, capsys
) -> None:
    """Dense APHI=0 plus APHI(2) is finite in VMEC2000 and must be in VMEX."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    path = inp.to_indata(tmp_path / "input.indexed_aphi")
    text = path.read_text().replace(
        "  APHI = 1.0000000000000000E+00",
        "  APHI = 0.0\n  APHI(2) = 1.0",
    )
    path.write_text(text)

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "R/Z force normalization valid: PASS" in output
    assert "lambda force normalization valid: PASS" in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output


def test_indexed_aphi_multivalue_prevents_false_zero_flux_diagnosis(
    tmp_path: Path, capsys
) -> None:
    """VMEC2000 treats APHI(1)=0,1 as two vector elements."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    path = inp.to_indata(tmp_path / "input.indexed_aphi_multivalue")
    text = path.read_text().replace(
        "  APHI = 1.0000000000000000E+00",
        "  APHI(1) = 0.0, 1.0",
    )
    path.write_text(text)

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "R/Z force normalization valid: PASS" in output
    assert "lambda force normalization valid: PASS" in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output


def test_shareable_diagnostic_accepts_compact_boundary_section(
    tmp_path: Path, capsys
) -> None:
    """A legal ``RBC(nlo:nhi,m)`` section reaches the first force pass."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    path = inp.to_indata(tmp_path / "input.private_boundary_section")
    text = path.read_text().replace(
        "&INDATA",
        "&INDATA\n  RBC(-6:6,0) = 13*0.0",
    )
    path.write_text(text)

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "private_boundary_section" not in output
    assert "RBC" not in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output


def test_first_force_uses_values_from_compact_boundary_sections(
    tmp_path: Path, capsys
) -> None:
    """Section-only boundary coefficients produce a finite equilibrium state."""
    path = tmp_path / "input.private_section_only"
    path.write_text(
        """&INDATA
        NFP = 1
        MPOL = 3
        NTOR = 1
        NS_ARRAY = 7
        RBC(-1:1,0) = 0.0, 1.0, 0.0
        RBC(-1:1,1) = 0.0, 0.1, 0.0
        ZBS(-1:1,1) = 0.0, 0.1, 0.0
        /
        """
    )

    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "private_section_only" not in output
    assert "RBC" not in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output
