"""Environment diagnostics for vmex installations."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import site
import subprocess
import sys
from importlib import metadata

from packaging.version import InvalidVersion, Version


_CORE_PACKAGES = (
    "vmex",
    "numpy",
    "jax",
    "jaxlib",
    "scipy",
    "netCDF4",
    "matplotlib",
    "booz_xform_jax",
    "setuptools",
    "packaging",
    "pip",
)


@dataclass(frozen=True)
class DoctorReport:
    """Structured environment report returned by :func:`collect_report`."""

    python: str
    executable: str
    prefix: str
    base_prefix: str
    platform: str
    in_virtualenv: bool
    conda_prefix: str | None
    user_site: str | None
    user_site_on_path: bool
    pip_report: str
    versions: dict[str, str]
    jax_backend: str | None
    jax_default_device: str | None
    jax_devices: tuple[str, ...]
    warnings: tuple[str, ...]


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:  # pragma: no cover - defensive diagnostics only.
        return f"error: {exc}"


def _version_at_least(version_text: str, minimum: str) -> bool:
    try:
        return Version(version_text) >= Version(minimum)
    except (InvalidVersion, TypeError):
        return False


def _pip_report() -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - depends on local interpreter.
        return f"unavailable: {exc}"
    return (proc.stdout or proc.stderr or "").strip() or f"pip exited with code {proc.returncode}"


def _user_site() -> str | None:
    try:
        return site.getusersitepackages()
    except Exception:
        return None


def _jax_info() -> tuple[str | None, tuple[str, ...], str | None]:
    try:
        import jax

        backend = str(jax.default_backend())
        devices = tuple(str(device) for device in jax.devices())
        return backend, devices, None
    except Exception as exc:
        return None, (), str(exc)


def _jax_default_device() -> str | None:
    try:
        import jax

        device = jax.config.jax_default_device
        return None if device is None else str(device)
    except Exception:  # pragma: no cover - defensive diagnostics only.
        return None


def collect_report() -> DoctorReport:
    """Collect installation diagnostics without modifying the environment."""
    versions = {name: _package_version(name) for name in _CORE_PACKAGES}
    user_site = _user_site()
    pip_text = _pip_report()
    in_virtualenv = sys.prefix != sys.base_prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    user_site_on_path = bool(user_site and user_site in sys.path)
    backend, devices, jax_error = _jax_info()

    warnings: list[str] = []
    if versions["setuptools"] == "not installed":
        warnings.append("setuptools is not installed; source/editable installs need it.")
    if versions["packaging"] == "not installed":
        warnings.append("packaging is not installed; source/editable installs may need it.")
    elif not _version_at_least(versions["packaging"], "24.2"):
        warnings.append("packaging may be too old for current setuptools license validation.")
    if versions["pip"] == "not installed":
        warnings.append("pip is not installed in this interpreter.")
    if user_site_on_path and not in_virtualenv and conda_prefix is None:
        warnings.append(
            "user-site packages are on sys.path outside a virtual environment; "
            "this can mix Homebrew/system packages with user installs."
        )
    pip_prefix_matches = sys.prefix in pip_text or bool(conda_prefix and conda_prefix in pip_text)
    if " from " in pip_text and not pip_prefix_matches:
        warnings.append("pip appears to come from a different prefix than the active Python environment.")
    if jax_error is not None:
        warnings.append(f"JAX import/backend check failed: {jax_error}")

    return DoctorReport(
        python=sys.version.replace("\n", " "),
        executable=sys.executable,
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        in_virtualenv=in_virtualenv,
        conda_prefix=conda_prefix,
        user_site=user_site,
        user_site_on_path=user_site_on_path,
        pip_report=pip_text,
        versions=versions,
        jax_backend=backend,
        jax_default_device=_jax_default_device(),
        jax_devices=devices,
        warnings=tuple(warnings),
    )


def format_report(report: DoctorReport) -> str:
    """Format a :class:`DoctorReport` for terminal output."""
    lines = [
        "vmex installation doctor",
        "----------------------------",
        f"Python:      {report.python}",
        f"Executable:  {report.executable}",
        f"Prefix:      {report.prefix}",
        f"Base prefix: {report.base_prefix}",
        f"Platform:    {report.platform}",
        f"Virtualenv:  {'yes' if report.in_virtualenv else 'no'}",
    ]
    if report.conda_prefix:
        lines.append(f"Conda env:   {report.conda_prefix}")
    lines.extend(
        [
            f"User site:   {report.user_site or 'unavailable'}",
            f"User site on sys.path: {'yes' if report.user_site_on_path else 'no'}",
            f"pip:         {report.pip_report}",
            "",
            "Packages:",
        ]
    )
    width = max(len(name) for name in report.versions)
    for name in _CORE_PACKAGES:
        lines.append(f"  {name:<{width}}  {report.versions.get(name, 'not checked')}")
    lines.extend(
        [
            "",
            f"JAX backend: {report.jax_backend or 'unavailable'}",
            f"JAX default device: {report.jax_default_device or 'automatic'}",
            "JAX devices:",
        ]
    )
    if report.jax_devices:
        lines.extend(f"  - {device}" for device in report.jax_devices)
    else:
        lines.append("  - none detected")
    lines.extend(
        [
            "VMEX forward default:  automatic work/mode-based CPU/GPU policy",
            "VMEX implicit default: CPU on accelerator hosts (explicit device= overrides)",
            "VMEX mirror default:   CPU for host-SciPy/JAX callback solves",
        ]
    )
    lines.append("")
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in report.warnings)
        lines.extend(
            [
                "",
                "Recommended clean install:",
                "  pip install vmex",
                "  If pip targets a different Python, use the matching python -m pip form.",
            ]
        )
    else:
        lines.append("Status: no obvious installation problems detected.")
    return "\n".join(lines)


def main() -> int:
    """Run the installation doctor CLI."""
    print(format_report(collect_report()))
    return 0
