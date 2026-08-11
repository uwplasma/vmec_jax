"""Bounded physics acceptance for optimizer-neutral QI and QS problems."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

from scipy.optimize import least_squares, minimize  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vmex.core import optimize as opt  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.omnigenity import QIResidual  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _seed(nfp: int, *, kick: float = 0.0) -> VmecInput:
    """Small converged seed: enough physics for descent, bounded CI cost."""
    inp = VmecInput.from_file(DATA / f"input.minimal_seed_nfp{nfp}")
    inp = inp.change_resolution(mpol=3, ntor=3, ntheta=12, nzeta=8)
    inp = dataclasses.replace(
        inp, ns_array=[13], ftol_array=[1.0e-9], niter_array=[800], delt=0.5,
    )
    if kick:
        rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
        rbc[inp.ntor + 1, 1] += kick
        zbs[inp.ntor + 1, 1] += kick
        inp = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
    return inp


def _run(problem):
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=4, ftol=1.0e-8, xtol=1.0e-10,
    )
    evaluation = problem.evaluate(result.x)
    assert evaluation.success
    assert evaluation.diagnostics["derivative_certified"]
    assert evaluation.diagnostics["derivative_fallbacks"] <= result.njev
    return problem.equilibrium_from_x(result.x)


def _run_scalar(problem):
    result = minimize(
        problem.value_and_grad, problem.x0, jac=True, method="L-BFGS-B",
        options={"maxiter": 3, "maxls": 6},
    )
    evaluation = problem.evaluate(result.x)
    assert evaluation.success
    assert evaluation.diagnostics["derivative_certified"]
    return problem.equilibrium_from_x(result.x)


@pytest.mark.full
def test_qi_implicit_descends():
    """QI residual descends without an optimizer-specific VMEX driver."""
    inp = _seed(2)
    qi = QIResidual(np.asarray([0.25, 0.6, 0.9]))
    problem = opt.VmecProblem.from_tuples(
        inp, [(qi, 0.0, 1.0), (opt.aspect_ratio, 6.0, 0.01)],
        max_mode=1, use_ess=True,
    )
    seed = float(qi.total(problem.equilibrium_from_x(problem.x0)))
    final = float(qi.total(_run(problem)))
    assert final < 0.95 * seed, f"QI residual {seed:.3e} -> {final:.3e}"


@pytest.mark.full
@pytest.mark.parametrize(
    "family,nfp,helicity,aspect,kick",
    [pytest.param("qa", 2, (1, 0), 6.0, 0.01, id="qa"),
     pytest.param("qh", 4, (1, -1), 8.0, 0.0, id="qh"),
     pytest.param("qp", 2, (0, 1), 6.0, 0.0, id="qp")],
)
def test_qs_implicit_descends(family, nfp, helicity, aspect, kick):
    """Each supported QS family has finite derivatives and physical descent."""
    inp = _seed(nfp, kick=kick)
    qs = opt.QuasisymmetryRatioResidual(np.asarray([0.4, 0.9]), *helicity)

    def loss(state, runtime):
        rows = qs.residuals_state(state, runtime)
        aspect_error = opt.aspect_ratio(state, runtime) - aspect
        return 0.5 * jnp.vdot(rows, rows) + 0.005 * aspect_error ** 2

    problem = opt.VmecProblem.from_loss(inp, loss, max_mode=1, use_ess=True)
    seed = float(qs.total(problem.equilibrium_from_x(problem.x0)))
    final = float(qs.total(_run_scalar(problem)))
    assert final < 0.95 * seed, f"{family.upper()} residual {seed:.3e} -> {final:.3e}"
