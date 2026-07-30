"""Unit tests for :mod:`vmex.doctor` (the ``vmec --doctor`` engine).

The report must reflect the running interpreter truthfully: package
versions come from importlib metadata, the JAX backend/devices are live,
and each warning heuristic fires exactly on its documented condition.
"""

from __future__ import annotations

import dataclasses
import sys

from vmex import doctor


def test_version_at_least():
    assert doctor._version_at_least("24.2", "24.2")
    assert doctor._version_at_least("25.0", "24.2")
    assert not doctor._version_at_least("24.1", "24.2")
    assert not doctor._version_at_least("not installed", "24.2")


def test_package_version_known_and_unknown():
    assert doctor._package_version("pytest") != "not installed"
    assert doctor._package_version("definitely-not-a-real-package") == "not installed"


def test_collect_report_reflects_interpreter():
    report = doctor.collect_report()
    assert report.executable == sys.executable
    assert report.prefix == sys.prefix
    assert sys.version.split()[0] in report.python
    # this test suite runs with JAX importable
    assert report.jax_backend is not None
    assert report.jax_default_device is None or isinstance(report.jax_default_device, str)
    assert len(report.jax_devices) >= 1
    assert set(doctor._CORE_PACKAGES) == set(report.versions)
    assert report.versions["numpy"] != "not installed"
    assert "pip" in report.pip_report.lower() or "unavailable" in report.pip_report


def test_format_report_healthy_and_warning_paths():
    report = doctor.collect_report()
    healthy = dataclasses.replace(report, warnings=(), conda_prefix=None)
    text = doctor.format_report(healthy)
    assert "vmex installation doctor" in text
    assert "Status: no obvious installation problems detected." in text
    assert "JAX backend:" in text
    assert "JAX default device:" in text
    assert "VMEX forward default:" in text
    assert "VMEX implicit default:" in text
    assert "VMEX mirror default:" in text
    for name in doctor._CORE_PACKAGES:
        assert name in text

    warned = dataclasses.replace(
        report, warnings=("something is off",), conda_prefix="/opt/conda/envs/x",
        jax_devices=(),
    )
    text = doctor.format_report(warned)
    assert "Warnings:" in text
    assert "  - something is off" in text
    assert "Recommended clean install:" in text
    assert "Conda env:   /opt/conda/envs/x" in text
    assert "  - none detected" in text


def test_warning_heuristics(monkeypatch):
    # missing setuptools/packaging/pip and a failing JAX import must each warn
    monkeypatch.setattr(doctor, "_package_version", lambda name: "not installed")
    monkeypatch.setattr(doctor, "_jax_info", lambda: (None, (), "boom"))
    monkeypatch.setattr(doctor, "_pip_report", lambda: "pip 25.0 from /elsewhere/site-packages (python 3.12)")
    report = doctor.collect_report()
    joined = "\n".join(report.warnings)
    assert "setuptools is not installed" in joined
    assert "packaging is not installed" in joined
    assert "pip is not installed" in joined
    assert "JAX import/backend check failed: boom" in joined
    assert "different prefix" in joined
    assert report.jax_backend is None


def test_user_site_and_jax_info_failure_paths(monkeypatch):
    import site

    monkeypatch.setattr(site, "getusersitepackages",
                        lambda: (_ for _ in ()).throw(RuntimeError("no site")))
    assert doctor._user_site() is None

    # a broken jax module surfaces as (None, (), error-text)
    monkeypatch.setitem(sys.modules, "jax", object())
    backend, devices, err = doctor._jax_info()
    assert backend is None and devices == () and err


def test_old_packaging_and_user_site_warnings(monkeypatch):
    real = doctor._package_version
    monkeypatch.setattr(
        doctor, "_package_version",
        lambda name: "24.1" if name == "packaging" else real(name))
    # simulate a system interpreter with user-site on sys.path
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(doctor, "_user_site", lambda: sys.path[-1])
    report = doctor.collect_report()
    joined = "\n".join(report.warnings)
    assert "packaging may be too old" in joined
    assert "user-site packages are on sys.path outside a virtual environment" in joined


def test_main_prints_report_and_returns_zero(capsys):
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert "vmex installation doctor" in out
