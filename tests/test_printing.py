"""Unit tests for :mod:`vmex.core.printing` (printout.f format port).

The screen/threed1 iteration lines and headers are asserted byte-for-byte
for representative values in every variant (symmetric/lasym, fixed/free
boundary), plus the stage/vacuum banners and the termination summary
(known and unknown ier_flag codes).
"""

from __future__ import annotations

from vmex.core import errors, printing


def test_stage_banner_format():
    s = printing.stage_banner(51, 137, 1e-14, 20000)
    assert s == "\n  NS =   51 NO. FOURIER MODES =  137 FTOLV =  1.000E-14 NITER =  20000\n"


def test_vacuum_banner_format():
    assert printing.vacuum_banner(38) == "\n  VACUUM PRESSURE TURNED ON AT   38 ITERATIONS\n"


def test_force_iterations_banner():
    assert "BEGIN FORCE ITERATIONS" in printing.FORCE_ITERATIONS_BANNER


def test_screen_header_variants():
    sym_fixed = printing.screen_header()
    assert sym_fixed == (
        "\n  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD\n"
    )
    assert "ZAX(v=0)" not in sym_fixed

    lasym = printing.screen_header(lasym=True)
    assert " ZAX(v=0)  " in lasym

    freeb = printing.screen_header(lfreeb=True)
    assert freeb.rstrip("\n").endswith("WMHD      DEL-BSQ")

    both = printing.screen_header(lasym=True, lfreeb=True)
    assert "ZAX(v=0)" in both and "DEL-BSQ" in both


def test_screen_line_fixed_symmetric():
    line = printing.screen_line(1, 9.99e-1, 8.88e-2, 7.77e-3, 3.999, 0.9, 1.4123e-1)
    assert line == "    1  9.99E-01  8.88E-02  7.77E-03  3.999E+00  9.00E-01  1.4123E-01\n"


def test_screen_line_lasym_and_freeb_columns():
    base = printing.screen_line(200, 1e-14, 2e-15, 3e-16, 3.999, 0.9, 0.14)
    lasym = printing.screen_line(200, 1e-14, 2e-15, 3e-16, 3.999, 0.9, 0.14,
                                 z_axis=-1.234e-2)
    assert len(lasym) == len(base) + 11
    assert " -1.234E-02" in lasym

    freeb = printing.screen_line(200, 1e-14, 2e-15, 3e-16, 3.999, 0.9, 0.14,
                                 del_bsq=5.5e-4)
    assert len(freeb) == len(base) + 11
    assert freeb.rstrip("\n").endswith("5.500E-04")


def test_threed1_header_variants():
    fixed = printing.threed1_header()
    assert fixed.startswith("\n  ITER    FSQR")
    assert fixed.endswith("<M>\n\n")
    freeb = printing.threed1_header(lfreeb=True)
    assert freeb.endswith("DEL-BSQ   FEDGE\n\n")


def test_threed1_line_fixed():
    line = printing.threed1_line(
        500, 1.0e-14, 2.0e-15, 3.0e-16, 4.0e-11, 5.0e-12, 6.0e-13,
        0.9, 3.999, 0.14123, 1.234e-2, 1.567,
    )
    assert line == (
        "   500   1.00E-14  2.00E-15  3.00E-16  4.00E-11  5.00E-12  6.00E-13"
        "  9.00E-01  3.999E+00  1.4123E-01  1.234E-02  1.567\n"
    )


def test_threed1_line_freeb_appends_vacuum_diagnostics():
    fixed = printing.threed1_line(1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                  0.9, 3.999, 0.14, 1e-2, 1.5)
    freeb = printing.threed1_line(1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                  0.9, 3.999, 0.14, 1e-2, 1.5,
                                  del_bsq=1.2e-3, f_edge=4.5e-6)
    assert len(freeb) == len(fixed) + 18
    assert freeb.rstrip("\n").endswith(" 1.20E-03 4.50E-06")
    # both diagnostics are required for the vacuum columns
    partial = printing.threed1_line(1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                    0.9, 3.999, 0.14, 1e-2, 1.5, del_bsq=1.2e-3)
    assert partial == fixed


def test_termination_summary_known_and_unknown_flags():
    s = printing.termination_summary(errors.SUCCESSFUL_TERM_FLAG, "input.solovev", 2, 12.5)
    assert "EXECUTION TERMINATED NORMALLY" in s
    assert " FILE : input.solovev\n" in s
    assert " NUMBER OF JACOBIAN RESETS =    2\n" in s
    assert "TOTAL COMPUTATIONAL TIME (SEC)        12.50" in s

    s = printing.termination_summary(errors.MORE_ITER_FLAG, "input.x", 0, 0.0)
    assert "MORE ITERATIONS REQUIRED" in s

    s = printing.termination_summary(999, "input.x", 0, 0.0)
    assert "UNKNOWN TERMINATION CODE" in s
