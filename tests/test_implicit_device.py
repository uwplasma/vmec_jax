"""Focused placement tests for the public implicit-differentiation API."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from vmex.core import implicit as im
from vmex.core import optimize as opt
from vmex.core.device import AUTO
from vmex.core.input import VmecInput


class _Stop(Exception):
    pass


def _device(kind):
    try:
        return jax.devices(kind)[0]
    except RuntimeError:
        pytest.skip(f"{kind.upper()} unavailable")


def _platforms(params):
    def platform(array):
        device = array.device
        return (device() if callable(device) else device).platform

    return {platform(leaf) for leaf in jax.tree.leaves(params)}


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_params_from_input_honors_explicit_device(kind):
    params = im.params_from_input(VmecInput(), device=_device(kind))
    assert _platforms(params) == {kind}


@pytest.mark.parametrize("requested", [AUTO, None, "cpu"])
def test_run_forwards_device_when_constructing_params(monkeypatch, requested):
    seen = []

    def params_from_input(inp, *, device=None):
        seen.append(device)
        raise _Stop

    monkeypatch.setattr(im, "params_from_input", params_from_input)
    with pytest.raises(_Stop):
        im.run(VmecInput(), device=requested)
    assert seen == [requested]


def test_run_preserves_supplied_params_for_auto_and_none(monkeypatch):
    params = im.params_from_input(VmecInput(), device="cpu")

    def solve_implicit(got, cfg):
        assert got is params
        raise _Stop

    monkeypatch.setattr(im, "solve_implicit", solve_implicit)
    for requested in (AUTO, None):
        with pytest.raises(_Stop):
            im.run(VmecInput(), params, device=requested)


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_run_places_supplied_params_on_requested_device(monkeypatch, kind):
    requested = _device(kind)
    inp = VmecInput()
    params = im.params_from_input(inp, device="cpu")

    def solve_implicit(got, cfg):
        assert _platforms(got) == {kind}
        raise _Stop

    monkeypatch.setattr(im, "solve_implicit", solve_implicit)
    with pytest.raises(_Stop):
        im.run(inp, params, device=requested)


def test_least_squares_places_params_on_jacobian_device(monkeypatch):
    requested = _device("cpu")
    seen = []

    def params_from_input(inp, *, device=None):
        seen.append(device)
        raise _Stop

    monkeypatch.setattr(im, "params_from_input", params_from_input)
    with pytest.raises(_Stop):
        opt.least_squares(
            [(opt.aspect_ratio, 4.0, 1.0)], VmecInput(), jac="implicit",
            device=requested,
        )
    assert seen == [requested]


def test_second_device_placement_and_gradient_no_outer_context(tmp_path):
    """Full-fidelity second-device audit on forced host devices (subprocess —
    the device-count flag must precede JAX init).  cpu:1 stands in for the
    second accelerator; caches are warmed by a default-device pass first
    (the poisoning order), then with NO outer context the second-device pass
    asserts: every state/runtime/gradient leaf committed to device 1
    (identity, not platform); ``value_and_grad`` matches the default device
    and is nonzero; a derived diagnostic stays finite; ``device="auto"``
    preserves the supplied parameters' committed home."""
    import os
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "second_device_audit.py"
    script.write_text(textwrap.dedent("""
        import jax
        import numpy as np
        import vmex
        print("VMEX-AT", __import__("pathlib").Path(vmex.__file__).resolve().parents[1])
        from vmex.core import implicit as im
        from vmex.core import optimize
        from vmex.core.input import VmecInput

        inp = VmecInput.from_file("examples/data/input.solovev")
        dev0, dev1 = jax.devices("cpu")[:2]

        def strays(tree, home):
            bad = set()
            for leaf in jax.tree.leaves(tree):
                if hasattr(leaf, "devices"):
                    ds = {str(d) for d in leaf.devices()}
                    if ds != {str(home)}:
                        bad |= ds
            return bad

        # default-device pass FIRST: warms every device-blind cache the way
        # a CPU-then-accelerator audit does
        p0 = im.params_from_input(inp, device=dev0)
        v0, g0 = jax.value_and_grad(lambda p: im.run(
            inp, p, ftol=1e-11, max_iterations=600, device=dev0).wb)(p0)

        p1 = im.params_from_input(inp, device=dev1)
        v1, g1 = jax.value_and_grad(lambda p: im.run(
            inp, p, ftol=1e-11, max_iterations=600, device=dev1).wb)(p1)
        s1 = im.run(inp, p1, ftol=1e-11, max_iterations=600, device=dev1)
        well = float(np.asarray(optimize.magnetic_well(s1.state, s1.runtime)))
        s1a = im.run(inp, p1, ftol=1e-11, max_iterations=600)  # AUTO

        assert not strays(s1.state, dev1), strays(s1.state, dev1)
        assert not strays(s1.runtime, dev1), strays(s1.runtime, dev1)
        assert not strays(g1, dev1), strays(g1, dev1)
        assert not strays(s1a.state, dev1) and not strays(s1a.runtime, dev1)
        gr0 = float(np.asarray(g0.rbc)[inp.ntor, 1])
        gr1 = float(np.asarray(g1.rbc)[inp.ntor, 1])
        assert gr1 != 0.0, "second-device gradient collapsed to zero"
        np.testing.assert_allclose(gr1, gr0, rtol=1e-9)
        np.testing.assert_allclose(float(v1), float(v0), rtol=1e-12)
        assert np.isfinite(well), "derived diagnostic went non-finite"
        print("SECOND-DEVICE AUDIT OK")
    """))
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + " --xla_force_host_platform_device_count=2")
    env["JAX_ENABLE_X64"] = "1"
    repo = str(Path(__file__).resolve().parents[1])
    # `python script.py` puts the SCRIPT's directory (tmp) on sys.path, not
    # the cwd — without this the child imports whatever vmex happens to be
    # installed (review found it picking up an older editable checkout).
    # Prepend this exact checkout and assert the child really used it.
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=repo, env=env,
        # 1800: the audit subprocess cold-imports jax and compiles the
        # implicit solve from scratch; ~25 s on a developer machine but
        # measured >600 s on a busy 4-core hosted runner sharing the shard
        # with three coverage workers (two c1 failures were exactly this
        # cap expiring).  The generous cap only bounds a genuine hang.
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "SECOND-DEVICE AUDIT OK" in proc.stdout
    assert f"VMEX-AT {repo}" in proc.stdout, proc.stdout[-500:]
