"""Regression tests for benchmark result classification."""

from benchmarks.run_baseline import parse_vmec2000


def test_vmec2000_niter_exhaustion_is_not_reported_as_convergence():
    result = parse_vmec2000({
        "ok": True,
        "stdout": """
 1000  1.29E-01  1.09E-01  3.95E-02
 Try increasing NITER or PRE_NITER if the preconditioner is on.
 EXECUTION TERMINATED NORMALLY
""",
    })

    assert result["ok"]
    assert not result["converged"]
    assert result["iterations"] == 1000
