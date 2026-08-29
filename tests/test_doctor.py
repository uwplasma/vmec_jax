"""Unit tests for :mod:`vmex.doctor` (the ``vmec --doctor`` engine).

The report must reflect the running interpreter truthfully: package
versions come from importlib metadata, the JAX backend/devices are live,
and each warning heuristic fires exactly on its documented condition.
"""

from __future__ import annotations

import dataclasses
import sys
import types

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
    assert report.jax_probe is not None and report.jax_probe.startswith("passed on ")
    assert isinstance(report.wsl2, bool)
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
    assert "JAX JIT probe:" in text
    assert "WSL2:" in text
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
    monkeypatch.setattr(doctor, "_jax_info", lambda: (None, (), None, "boom"))
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
    backend, devices, probe, err = doctor._jax_info()
    assert backend is None and devices == () and probe is None and err


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


def test_wsl2_gpu_reports_known_jaxlib_fixes(monkeypatch):
    real_version = doctor._package_version
    monkeypatch.setattr(doctor, "_is_wsl2", lambda: True)
    monkeypatch.setattr(
        doctor,
        "_package_version",
        lambda name: "0.9.2" if name in ("jax", "jaxlib") else real_version(name),
    )
    monkeypatch.setattr(
        doctor,
        "_nvidia_smi",
        lambda: ("NVIDIA GeForce RTX 4090, 566.36", None),
    )
    monkeypatch.setattr(
        doctor,
        "_jax_info",
        lambda: ("gpu", ("cuda:0",), "passed on cuda:0 (0.100 s)", None),
    )
    report = doctor.collect_report()
    joined = "\n".join(report.warnings)
    assert report.wsl2
    assert report.nvidia_smi == "NVIDIA GeForce RTX 4090, 566.36"
    assert "0.10.1 or newer" in joined
    assert "PJRT" in joined
    assert "two-component Windows NVIDIA driver versions" in joined
    text = doctor.format_report(report)
    assert "WSL2:        yes" in text
    assert "WSL2 NVIDIA: NVIDIA GeForce RTX 4090, 566.36" in text


def test_wsl2_detection_and_nvidia_smi_fallback(monkeypatch):
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(doctor.platform, "release", lambda: "6.6.0-linux")
    monkeypatch.setattr(doctor.platform, "version", lambda: "plain Linux")
    assert not doctor._is_wsl2()
    monkeypatch.setattr(
        doctor.platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2"
    )
    assert doctor._is_wsl2()

    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=0,
            stdout="NVIDIA RTX A6000, 572.61\n",
            stderr="",
        ),
    )
    summary, error = doctor._nvidia_smi()
    assert summary == "NVIDIA RTX A6000, 572.61"
    assert error is None

    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor.os.path, "isfile", lambda path: False)
    summary, error = doctor._nvidia_smi()
    assert summary is None and "not found" in error


def test_nvidia_smi_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="driver unavailable",
        ),
    )
    summary, error = doctor._nvidia_smi()
    assert summary is None
    assert error == "nvidia-smi failed: driver unavailable"


def test_nvidia_smi_uses_the_standard_wsl_fallback(monkeypatch):
    observed = {}
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor.os.path, "isfile", lambda path: True)

    def run(args, **kwargs):
        observed["args"] = args
        return types.SimpleNamespace(
            returncode=0,
            stdout="NVIDIA RTX 4090, 566.36\n",
            stderr="",
        )

    monkeypatch.setattr(doctor.subprocess, "run", run)
    summary, error = doctor._nvidia_smi()
    assert observed["args"][0] == "/usr/lib/wsl/lib/nvidia-smi"
    assert summary == "NVIDIA RTX 4090, 566.36"
    assert error is None


def test_jax_probe_covers_empty_bad_and_legacy_device_paths(monkeypatch):
    import jax

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "devices", lambda: ())
    backend, devices, probe, error = doctor._jax_info()
    assert backend is None and devices == () and probe is None
    assert error == "JAX returned no devices"

    class FakeResult:
        def __init__(self, value):
            self.value = value

        def block_until_ready(self):
            return self

        def __float__(self):
            return float(self.value)

        def device(self):
            return "cuda:0"

    monkeypatch.setattr(jax, "devices", lambda: ("cuda:0",))
    monkeypatch.setattr(jax, "device_put", lambda values, device: values)
    monkeypatch.setattr(jax, "device_get", lambda result: result)
    monkeypatch.setattr(jax, "jit", lambda function: lambda values: FakeResult(14.0))
    backend, devices, probe, error = doctor._jax_info()
    assert backend == "gpu" and devices == ("cuda:0",)
    assert probe.startswith("passed on cuda:0") and error is None

    monkeypatch.setattr(jax, "jit", lambda function: lambda values: FakeResult(13.0))
    backend, devices, probe, error = doctor._jax_info()
    assert backend is None and devices == () and probe is None
    assert error == "JIT device probe returned 13.0, expected 14.0"


def test_wsl2_visible_gpu_but_cpu_jax_warns(monkeypatch):
    monkeypatch.setattr(doctor, "_is_wsl2", lambda: True)
    monkeypatch.setattr(doctor, "_nvidia_smi", lambda: ("NVIDIA RTX, 580.1", None))
    monkeypatch.setattr(
        doctor,
        "_jax_info",
        lambda: ("cpu", ("TFRT_CPU_0",), "passed on TFRT_CPU_0 (0.010 s)", None),
    )
    report = doctor.collect_report()
    assert any("JAX selected the cpu backend" in item for item in report.warnings)


def test_current_wsl2_gpu_does_not_warn_about_fixed_jaxlib(monkeypatch):
    real_version = doctor._package_version
    monkeypatch.setattr(doctor, "_is_wsl2", lambda: True)
    monkeypatch.setattr(
        doctor,
        "_package_version",
        lambda name: "0.10.1" if name in ("jax", "jaxlib") else real_version(name),
    )
    monkeypatch.setattr(doctor, "_nvidia_smi", lambda: (None, "driver query failed"))
    monkeypatch.setattr(
        doctor,
        "_jax_info",
        lambda: ("gpu", ("cuda:0",), "passed on cuda:0 (0.100 s)", None),
    )
    report = doctor.collect_report()
    joined = "\n".join(report.warnings)
    assert "0.10.1 or newer" not in joined
    assert "driver query failed" in joined


def test_compilation_cache_line_reports_each_cache_state(monkeypatch, tmp_path):
    """The cache line distinguishes disabled, not-yet-created, and in-use.

    A cache sitting at its bound is the visible symptom of eviction churn, so
    the line must name the bound and flag it, and it must never raise when the
    directory is absent or the feature is switched off.
    """
    from vmex import _compat

    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setattr(_compat, "_default_compilation_cache_dir", lambda: "")
    assert doctor._compilation_cache_line().endswith("disabled")

    missing = tmp_path / "absent"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(missing))
    assert "not yet created" in doctor._compilation_cache_line()

    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path))
    (tmp_path / "entry.bin").write_bytes(b"x" * 4096)
    monkeypatch.setattr(_compat, "_default_cache_max_size", lambda _dir: 4096.0)
    line = doctor._compilation_cache_line()
    assert "GiB" in line and "at the bound" in line
