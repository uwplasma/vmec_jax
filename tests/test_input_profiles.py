"""Input parsing and writer round-trip tests for ``vmex.core.input``.

For every bundled input deck, :class:`vmex.core.input.VmecInput` is
round-tripped through both writers (JSON and INDATA) and must reproduce every
field exactly.  A VMEC++ JSON example (``data/solovev.json``, copied verbatim
from the vmecpp repository) validates JSON-schema compatibility.  (Field-level
parity with the historical parser stack was proven by the A/B suite that
accompanied the port and retired with the legacy tree.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmex.core.input import UnsupportedInputModeError, VmecInput

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
DECKS = sorted(DATA.glob("input.*"))
FIXTURES = Path(__file__).parent / "data"

assert DECKS, f"no input decks found under {DATA}"


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_json_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_json -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_json(tmp_path / "roundtrip.json"))
    assert reread == original


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_indata_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_indata -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_indata(tmp_path / "input.roundtrip"))
    assert reread == original


def test_vmecpp_json_example() -> None:
    """The VMEC++ solovev.json example parses with VMEC++ schema semantics."""
    new = VmecInput.from_file(FIXTURES / "solovev.json")
    assert new.mpol == 6
    assert new.ntor == 0
    assert new.nfp == 1  # default
    assert new.lasym is False  # default
    assert new.ncurr == 0
    assert new.nstep == 250
    assert new.delt == 0.9
    np.testing.assert_array_equal(new.ns_array, [5, 11, 55])
    np.testing.assert_array_equal(new.niter_array, [1000, 2000, 2000])
    np.testing.assert_array_equal(new.ftol_array, [1e-16, 1e-16, 1e-16])
    np.testing.assert_array_equal(new.am[:3], [0.125, -0.125, 0.0])
    assert not np.any(new.am[3:])
    np.testing.assert_array_equal(new.ai[:2], [1.0, 0.0])
    np.testing.assert_array_equal(new.raxis_c, [4.0])
    np.testing.assert_array_equal(new.zaxis_s, [0.0])
    # Sparse boundary lists land at [n + ntor, m] = [0, m] for ntor = 0.
    np.testing.assert_array_equal(new.rbc[0, :3], [3.999, 1.026, -0.068])
    np.testing.assert_array_equal(new.zbs[0, :3], [0.0, 1.58, 0.01])
    assert not np.any(new.rbs) and not np.any(new.zbc)
    assert new.lfreeb is False  # VMEC++ default
    assert new.gamma == 0.0  # default (JSON alias: adiabatic_index)
    assert new.mgrid_file == "NONE"
    assert new.precon_type == "NONE"
    assert new.lmove_axis is True

    # And it survives both writers.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        assert VmecInput.from_file(new.to_json(Path(tmp) / "x.json")) == new
        assert VmecInput.from_file(new.to_indata(Path(tmp) / "input.x")) == new


def test_lmove_axis_default_explicit_false_and_indata_round_trip(
    tmp_path: Path,
) -> None:
    """VMEC2000 defaults LMOVE_AXIS=T and permits an explicit opt-out."""
    assert VmecInput.from_indata_text("&INDATA\n/\n").lmove_axis is True
    inp = VmecInput.from_indata_text("&INDATA\nLMOVE_AXIS = F\n/\n")
    assert inp.lmove_axis is False
    assert VmecInput.from_file(
        inp.to_indata(tmp_path / "input.lmove_axis")
    ).lmove_axis is False


def test_lforbal_default_explicit_true_and_indata_round_trip(
    tmp_path: Path,
) -> None:
    """VMEC2000 defaults LFORBAL=F and supports the active replacement."""
    assert VmecInput.from_indata_text("&INDATA\n/\n").lforbal is False
    inp = VmecInput.from_indata_text("&INDATA\nLFORBAL = T\n/\n")
    assert inp.lforbal is True
    assert VmecInput.from_file(
        inp.to_indata(tmp_path / "input.lforbal")
    ).lforbal is True


def test_lfull3d1out_default_explicit_true_and_indata_round_trip(
    tmp_path: Path,
) -> None:
    """VMEC2000 defaults LFULL3D1OUT=F and permits an explicit WOUT request."""
    assert VmecInput.from_indata_text("&INDATA\n/\n").lfull3d1out is False
    inp = VmecInput.from_indata_text("&INDATA\nLFULL3D1OUT = T\n/\n")
    assert inp.lfull3d1out is True
    assert VmecInput.from_file(
        inp.to_indata(tmp_path / "input.lfull3d1out")
    ).lfull3d1out is True


def test_legacy_ns_array_zero_expands_via_nsin() -> None:
    """VMEC2000's explicit old-style sentinel builds the NSIN -> 31 ladder."""
    inp = VmecInput.from_indata_text(
        "&INDATA\nNS_ARRAY(1) = 0\nNSIN = 7\n/\n"
    )
    np.testing.assert_array_equal(inp.ns_array, [7, 31])


def test_legacy_niter_scalar_fills_the_multigrid_ladder() -> None:
    """NITER is the VMEC2000 fallback only when NITER_ARRAY is unassigned."""
    inp = VmecInput.from_indata_text(
        "&INDATA\nNS_ARRAY = 7, 15\nNITER = 321\n/\n"
    )
    np.testing.assert_array_equal(inp.niter_array, [321, 321])


def test_readin_stops_at_first_decreasing_ns_entry() -> None:
    """Later resolutions do not resume after readin.f closes multi_ns_grid."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        NS_ARRAY = 21, 34, 20, 55
        FTOL_ARRAY = 1e-7, 1e-8, 1e-9, 1e-10
        NITER_ARRAY = 100, 200, 300, 400
        /
        """
    )
    np.testing.assert_array_equal(inp.ns_array, [21, 34])
    np.testing.assert_array_equal(inp.ftol_array, [1e-7, 1e-8])
    np.testing.assert_array_equal(inp.niter_array, [100, 200])


def test_zero_first_ftol_builds_vmec2000_geometric_ladder() -> None:
    """readin.f interpolates from 1e-8 to scalar FTOL across the ladder."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        NS_ARRAY = 21, 34, 55, 89
        FTOL = 1e-14
        FTOL_ARRAY(1) = 0.0
        /
        """
    )
    expected = 1.0e-8 * (1.0e-6 ** (np.arange(4) / 3.0))
    np.testing.assert_allclose(inp.ftol_array, expected, rtol=1e-15)
    assert inp.ftol_array[-1] == pytest.approx(1e-14)


def test_partial_niter_array_preserves_unassigned_minus_one_entries() -> None:
    """Any explicit NITER_ARRAY write disables the all--1 scalar fallback."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        NS_ARRAY = 21, 34, 55
        NITER = 999
        NITER_ARRAY(2) = 400
        /
        """
    )
    np.testing.assert_array_equal(inp.niter_array, [-1, 400, -1])


def test_partial_niter_minus_one_executes_one_vmec_pass() -> None:
    """eqsolve evaluates once before its ``iter2 >= niter`` termination."""
    from vmex.core.solver import prepare_runtime, resolution_from_input

    inp = VmecInput.from_indata_text(
        """&INDATA
        NS_ARRAY = 7, 11
        NITER_ARRAY(2) = 400
        /
        """
    )
    runtime = prepare_runtime(inp, resolution_from_input(inp, ns=7))
    assert inp.niter_array[0] == -1
    assert runtime.max_iterations == 1


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("LRECON=T\nIMSE=1", "D00A_RECONSTRUCTION_MODE_UNSUPPORTED"),
        ("LRFP=T", "D00B_RFP_MODE_UNSUPPORTED"),
        ("TRIP3D_FILE='field.nc'", "D00E_TRIP3D_MODE_UNSUPPORTED"),
        ("AH(0)=0.1", "D00F_ANIMEC_MODE_UNSUPPORTED"),
        ("AT=0.9", "D00F_ANIMEC_MODE_UNSUPPORTED"),
        ("TVOLUME=10.0", "D00G_VOLUME_RESCALE_UNSUPPORTED"),
        ("PRECON_TYPE='CG'", "D00H_PRECONDITIONER_MODE_UNSUPPORTED"),
        (
            "PRECON_TYPE='GMRES'\nPRE_NITER=20",
            "D00I_ITERATION_CONTROL_UNSUPPORTED",
        ),
        ("MAX_MAIN_ITERATIONS=2", "D00I_ITERATION_CONTROL_UNSUPPORTED"),
        ("LGIVEUP=T", "D00I_ITERATION_CONTROL_UNSUPPORTED"),
        ("LBSUBS=T", "D00J_OUTPUT_MODE_UNSUPPORTED"),
        ("LNYQUIST=F", "D00J_OUTPUT_MODE_UNSUPPORTED"),
        ("LBOOZ=T", "D00J_OUTPUT_MODE_UNSUPPORTED"),
    ],
)
def test_active_unsupported_indata_modes_fail_in_production_parser(
    body: str, code: str,
) -> None:
    """A parsed control must never silently select an ordinary VMEX solve."""
    with pytest.raises(UnsupportedInputModeError) as caught:
        VmecInput.from_indata_text(f"&INDATA\n{body}\n/\n")
    assert caught.value.code == code


def test_neutral_legacy_modes_remain_accepted() -> None:
    """Default-valued legacy controls do not require deck surgery."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        LRECON = T
        IMSE = -1
        ITSE = 0
        LRFP = F
        TRIP3D_FILE = 'NONE'
        AH = 0.0
        AT = 1.0, 0.0
        TVOLUME = -1.0
        PRECON_TYPE = 'DEFAULT'
        PRE_NITER = -1
        MAX_MAIN_ITERATIONS = 1
        LGIVEUP = F
        LBSUBS = F
        LNYQUIST = T
        LBOOZ = F
        LDIAGNO = F
        /
        """
    )
    assert inp.precon_type == "DEFAULT"


def test_time_slice_reaches_vmec_style_run_header() -> None:
    """The informational VMEC header field must not be parsed and discarded."""
    from vmex.core.cli import _preamble

    inp = VmecInput.from_indata_text("&INDATA\nTIME_SLICE=1.25\n/\n")
    assert "TIME SLICE  1.2500E+00" in _preamble(
        "case", time_slice=inp.time_slice
    )


def test_unimplemented_legacy_artifact_request_warns() -> None:
    """An output-only compatibility flag is allowed but never silently ignored."""
    with pytest.warns(RuntimeWarning, match="LDIAGNO.*not implemented"):
        VmecInput.from_indata_text("&INDATA\nLDIAGNO=T\n/\n")


def test_vmecpp_json_modes_and_unknown_keys_are_not_ignored() -> None:
    """VMEC++ model selection and misspelled keys fail before setup."""
    with pytest.raises(
        UnsupportedInputModeError,
        match="free_boundary_method",
    ) as caught:
        VmecInput.from_json_text('{"free_boundary_method": "only_coils"}')
    assert caught.value.code == "D00K_FREE_BOUNDARY_METHOD_UNSUPPORTED"

    with pytest.raises(ValueError, match="unknown VMEC\\+\\+ JSON input key"):
        VmecInput.from_json_text('{"free_boundary_method": "nestor", "mpoll": 6}')


def test_unknown_indata_name_is_not_ignored() -> None:
    """A misspelled physics control cannot become an accidental default."""
    with pytest.raises(ValueError, match="unknown INDATA variable.*PHIEDG"):
        VmecInput.from_indata_text("&INDATA\nPHIEDG = 0.5\n/\n")


def test_indexed_indata_vectors_overlay_vmec_defaults() -> None:
    """Indexed 1-D assignments follow VMEC2000 namelist lower bounds/defaults."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(2) = 0.25
        AM(0) = 1000.0
        AI(1) = -0.125
        AC(2) = 0.5
        NS_ARRAY(1) = 7
        NS_ARRAY(2) = 15
        FTOL_ARRAY(2) = 1e-12
        NITER_ARRAY(2) = 400
        EXTCUR(2) = -3.0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )

    # vmec_input.f initializes APHI(1)=1 before applying APHI(2).
    np.testing.assert_array_equal(inp.aphi[:3], [1.0, 0.25, 0.0])
    np.testing.assert_array_equal(inp.am[:3], [1000.0, 0.0, 0.0])
    np.testing.assert_array_equal(inp.ai[:3], [0.0, -0.125, 0.0])
    np.testing.assert_array_equal(inp.ac[:3], [0.0, 0.0, 0.5])
    np.testing.assert_array_equal(inp.ns_array, [7, 15])
    np.testing.assert_array_equal(inp.ftol_array, [1e-10, 1e-12])
    # A partial NITER_ARRAY assignment prevents VMEC2000's all--1 fallback;
    # unassigned active entries therefore remain -1.
    np.testing.assert_array_equal(inp.niter_array, [-1, 400])
    np.testing.assert_array_equal(inp.extcur, [0.0, -3.0])


def test_indexed_aphi_can_override_identity_term() -> None:
    """APHI(1) is one-based and may explicitly replace the VMEC default."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(1) = 0.5
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:3], [0.5, 0.0, 0.0])


def test_indexed_aphi_overlays_dense_assignment() -> None:
    """A sparse APHI term is not discarded after a dense leading zero."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI = 0.0
        APHI(2) = 1.0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:3], [0.0, 1.0, 0.0])


def test_indexed_aphi_starting_element_consumes_following_values() -> None:
    """Fortran ``APHI(1)=0,1`` assigns two elements, not one scalar."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(1) = 0.0, 1.0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:3], [0.0, 1.0, 0.0])


def test_indexed_aphi_section_uses_fortran_inclusive_bounds() -> None:
    """A non-leading array section retains VMEC's initialized first term."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(2:3) = 0.25, 0.5
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:4], [1.0, 0.25, 0.5, 0.0])


def test_boundary_section_accepts_toroidal_mode_range() -> None:
    """VMEC boundary rows accept compact ``RBC(nlo:nhi,m)`` assignments."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 2
        NTOR = 2
        RBC(-2:2,0) = 1.0, 2.0, 3.0, 4.0, 5.0
        ZBS(-2:2,1) = -1.0, -2.0, -3.0, -4.0, -5.0
        /
        """
    )
    np.testing.assert_array_equal(inp.rbc[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(inp.zbs[:, 1], [-1.0, -2.0, -3.0, -4.0, -5.0])


def test_boundary_section_uses_fortran_column_major_order() -> None:
    """The first boundary subscript varies fastest, as in Fortran namelists."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 2
        NTOR = 1
        RBC(-1:1,0:1) = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        /
        """
    )
    np.testing.assert_array_equal(inp.rbc[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(inp.rbc[:, 1], [4.0, 5.0, 6.0])


def test_boundary_section_accepts_negative_stride() -> None:
    """Inclusive section bounds and negative strides follow Fortran syntax."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 2
        NTOR = 2
        RBC(2:-2:-2,0) = 1.0, 2.0, 3.0
        /
        """
    )
    np.testing.assert_array_equal(inp.rbc[:, 0], [3.0, 0.0, 2.0, 0.0, 1.0])


def test_negative_stride_section_uses_fortran_default_bounds() -> None:
    """Omitted bounds reverse the declared extent when the stride is negative."""
    values = ", ".join(str(value) for value in range(1, 21))
    inp = VmecInput.from_indata_text(
        f"""&INDATA
        APHI(::-1) = {values}
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:3], [20.0, 19.0, 18.0])


def test_repeated_dense_assignments_overlay_in_source_order() -> None:
    """A short later assignment preserves the earlier Fortran array tail."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        NS_ARRAY = 11, 25, 49, 75
        FTOL_ARRAY = 1e-7, 1e-8, 1e-9, 1e-10
        FTOL_ARRAY = 1e-6, 1e-11
        /
        """
    )
    np.testing.assert_array_equal(inp.ns_array, [11, 25, 49, 75])
    np.testing.assert_array_equal(inp.ftol_array, [1e-6, 1e-11, 1e-9, 1e-10])


def test_dense_and_indexed_assignments_honor_source_order() -> None:
    """Later dense writes override earlier indexed elements, not vice versa."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        APHI(2) = 0.25
        APHI = 0.0, 1.0
        APHI(3) = 0.5
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:4], [0.0, 1.0, 0.5, 0.0])


def test_multidimensional_starting_element_continues_in_fortran_order() -> None:
    """Multiple values after ``RBC(n,m)`` continue with n varying fastest."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 2
        NTOR = 2
        RBC(-1,0) = 1.0, 2.0, 3.0, 4.0
        /
        """
    )
    np.testing.assert_array_equal(inp.rbc[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("designator", ["RBC", "RBC(:,0)", "RBC(-101:,0)"])
def test_dense_and_open_boundary_sections_use_declared_fortran_bounds(
    designator: str,
) -> None:
    """Whole/open sections are resolved from VMEC2000's allocated bounds."""
    inp = VmecInput.from_indata_text(
        f"""&INDATA
        MPOL = 2
        NTOR = 1
        {designator} = 101*0.0, 2.5
        /
        """
    )
    assert inp.rbc[1, 0] == 2.5  # physical n=0 at row ntor
    assert np.count_nonzero(inp.rbc) == 1


def test_indexed_legacy_axis_overlays_vmec_axis() -> None:
    """Indexed obsolete RAXIS/ZAXIS entries retain VMEC2000 compatibility."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 1
        RAXIS_CC(0) = 1.0
        RAXIS(0) = 2.0
        RAXIS(1) = 0.25
        ZAXIS(1) = -0.5
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.raxis_c, [2.0, 0.25])
    np.testing.assert_array_equal(inp.zaxis_s, [0.0, -0.5])
