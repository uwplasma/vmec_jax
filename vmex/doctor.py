"""Environment diagnostics behind ``vmex --doctor``.

The module answers one question: *is this interpreter able to run VMEX, and
which accelerator will it use?*  It never installs, upgrades, or configures
anything — :func:`collect_report` only reads, and the two subprocesses it
starts (``python -m pip --version``, and ``nvidia-smi`` on WSL2 only) are
read-only queries with short timeouts.

Three pieces:

- :class:`DoctorReport` — the structured, immutable snapshot: interpreter
  identity and prefixes, virtualenv/conda/user-site layout, the versions of
  the packages in ``_CORE_PACKAGES``, the live JAX backend and device list
  with the result of a real jitted device probe, and the accumulated
  ``warnings``.
- :func:`collect_report` — builds one, applying the warning heuristics.
- :func:`format_report` / :func:`main` — render it as the fixed-width block
  the CLI prints.

The warning heuristics target the failure modes that actually strand a VMEX
install: a missing ``setuptools``/``packaging``/``pip`` (source and editable
installs need them), user-site packages leaking onto ``sys.path`` outside a
virtual environment, a ``pip`` that belongs to a different prefix than the
running interpreter, a JAX import or backend failure, and — on WSL2 with a
GPU backend — a ``jaxlib`` older than 0.10.1 or an ``nvidia-smi`` that
disagrees with the backend JAX chose.

An empty ``warnings`` tuple is the healthy state and prints as a single
status line.  Warnings are advisory only: :func:`main` returns ``0``
unconditionally, so ``vmex --doctor`` exits ``0`` even when it reports
problems — read the printed text, not the exit status.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import shutil
import site
import subprocess
import sys
from time import perf_counter
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
    """Immutable snapshot of one interpreter, as :func:`collect_report` saw it.

    Every field is observed, never assumed: an unavailable probe is recorded
    as ``None`` or as an explicit ``"not installed"`` / ``"unavailable: ..."``
    string rather than being guessed or silently defaulted.

    Attributes
    ----------
    python:
        ``sys.version`` with newlines flattened to spaces.
    executable:
        ``sys.executable`` — the interpreter that ran the check.
    prefix, base_prefix:
        ``sys.prefix`` and ``sys.base_prefix``.  They differ exactly when a
        virtual environment is active.
    platform:
        ``"<system> <release> <machine>"`` from :mod:`platform`.
    in_virtualenv:
        ``prefix != base_prefix``.
    conda_prefix:
        ``$CONDA_PREFIX``, or ``None`` when no conda environment is active.
    user_site:
        ``site.getusersitepackages()``, or ``None`` if it could not be
        resolved.
    user_site_on_path:
        Whether that directory is actually on ``sys.path``.
    pip_report:
        The one-line output of ``python -m pip --version`` for *this*
        interpreter, or a message explaining why it could not be obtained.
    versions:
        ``{distribution name: version}`` for ``_CORE_PACKAGES``
        (``vmex``, ``numpy``, ``jax``, ``jaxlib``, ``scipy``, ``netCDF4``,
        ``matplotlib``, ``booz_xform_jax``, ``setuptools``, ``packaging``,
        ``pip``), read from installed distribution metadata — not from
        importing the packages.  A missing distribution reads
        ``"not installed"``.
    wsl2:
        Whether this process is running under Windows Subsystem for Linux
        (detected from the kernel release/version string or
        ``$WSL_INTEROP``).
    nvidia_smi:
        ``"<gpu name>, <driver version>"`` rows joined by ``"; "``, collected
        only on WSL2.  ``None`` elsewhere, and ``None`` on WSL2 when the
        utility is absent or failed (the reason then appears in
        ``warnings``).
    jax_backend:
        ``jax.default_backend()`` — typically ``"cpu"``, ``"gpu"``/``"cuda"``,
        or ``"tpu"``.  ``None`` when JAX could not be imported or queried.
    jax_default_device:
        ``jax.config.jax_default_device`` as a string, or ``None`` when no
        device has been pinned (the usual case, printed as ``automatic``).
    jax_devices:
        String form of every device ``jax.devices()`` reports.
    jax_probe:
        Result of a live end-to-end check — ``jnp.arange(4)`` is placed on the
        first device, a jitted ``vdot`` is compiled and run, and the value is
        verified against the exact answer, ``14.0``.  Reads
        ``"passed on <device> (<seconds> s)"``, whose timing is dominated by
        JAX import plus that first compilation.  ``None`` if any step raised;
        the exception text is then in ``warnings``.
    warnings:
        Advisory messages, empty when nothing suspicious was found.  They do
        not affect the process exit status.
    """

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
    wsl2: bool
    nvidia_smi: str | None
    jax_backend: str | None
    jax_default_device: str | None
    jax_devices: tuple[str, ...]
    jax_probe: str | None
    warnings: tuple[str, ...]


def _compilation_cache_line() -> str:
    """Report the persistent compilation cache's directory, size, and bound.

    A cache sitting at its bound is the visible symptom of eviction churn:
    every run then recompiles what the previous one stored.
    """
    from ._compat import _default_cache_max_size, _default_compilation_cache_dir

    directory = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR") or _default_compilation_cache_dir()
    if not directory:
        return "VMEX compile cache:    disabled"
    try:
        used = sum(entry.stat().st_size for entry in os.scandir(directory)
                   if entry.is_file())
    except OSError:
        return f"VMEX compile cache:    {directory} (not yet created)"
    cap = _default_cache_max_size(directory)
    note = "  [at the bound: raise VMEX_COMPILATION_CACHE_MAX_SIZE]" if (
        used >= 0.95 * cap) else ""
    return (f"VMEX compile cache:    {used / 2**30:.2f} of "
            f"{cap / 2**30:.2f} GiB in {directory}{note}")


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


def _is_wsl2() -> bool:
    """Return whether this process is running under Windows Subsystem for Linux."""

    marker = f"{platform.release()} {platform.version()}".lower()
    return "microsoft" in marker or bool(os.environ.get("WSL_INTEROP"))


def _nvidia_smi() -> tuple[str | None, str | None]:
    """Return GPU/driver rows from the Windows-provided WSL NVIDIA utility."""

    executable = shutil.which("nvidia-smi")
    if executable is None and os.path.isfile("/usr/lib/wsl/lib/nvidia-smi"):
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        return None, "nvidia-smi was not found (also checked /usr/lib/wsl/lib)."
    try:
        proc = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - host utility failure.
        return None, f"nvidia-smi could not run: {exc}"
    output = (proc.stdout or "").strip()
    if proc.returncode != 0 or not output:
        detail = (proc.stderr or output or f"exit code {proc.returncode}").strip()
        return None, f"nvidia-smi failed: {detail}"
    return "; ".join(line.strip() for line in output.splitlines() if line.strip()), None


def _jax_info() -> tuple[str | None, tuple[str, ...], str | None, str | None]:
    try:
        import jax
        import jax.numpy as jnp

        started = perf_counter()
        backend = str(jax.default_backend())
        live_devices = tuple(jax.devices())
        devices = tuple(str(device) for device in live_devices)
        if not live_devices:
            raise RuntimeError("JAX returned no devices")
        values = jax.device_put(jnp.arange(4, dtype=jnp.float64), live_devices[0])
        result = jax.jit(lambda value: jnp.vdot(value, value))(values)
        result.block_until_ready()
        observed = float(jax.device_get(result))
        if abs(observed - 14.0) > 1.0e-12:
            raise RuntimeError(f"JIT device probe returned {observed}, expected 14.0")
        result_device = getattr(result, "device", live_devices[0])
        if callable(result_device):  # JAX releases before Array.device became a property.
            result_device = result_device()
        probe = f"passed on {result_device} ({perf_counter() - started:.3f} s)"
        return backend, devices, probe, None
    except Exception as exc:
        return None, (), None, str(exc)


def _jax_default_device() -> str | None:
    try:
        import jax

        device = jax.config.jax_default_device
        return None if device is None else str(device)
    except Exception:  # pragma: no cover - defensive diagnostics only.
        return None


def collect_report() -> DoctorReport:
    """Collect installation diagnostics without modifying the environment.

    Observes, in order: distribution versions for ``_CORE_PACKAGES``, the
    user-site directory and whether it is on ``sys.path``, ``pip --version``
    for this interpreter, the virtualenv/conda layout, whether the host is
    WSL2 (and only then, ``nvidia-smi``), and the live JAX backend, devices,
    and jitted device probe.  Nothing is installed, upgraded, or configured;
    the only side effects are the two short-lived read-only subprocesses and
    importing JAX.

    It then applies the warning heuristics, each of which fires on exactly one
    condition: ``setuptools`` missing; ``packaging`` missing or older than
    24.2; ``pip`` missing; user-site on ``sys.path`` while neither a
    virtualenv nor a conda environment is active; a ``pip --version`` whose
    reported location matches neither ``sys.prefix`` nor ``$CONDA_PREFIX``;
    the JAX probe raising; and, on WSL2 with a GPU backend, ``jaxlib`` older
    than 0.10.1 or a failing ``nvidia-smi``.  One further heuristic fires the
    other way round: on WSL2, ``nvidia-smi`` seeing a GPU while JAX selected a
    non-GPU backend.

    Never raises for a broken environment — a failure becomes a field value
    and a warning string, which is the point of a doctor.

    Returns
    -------
    The :class:`DoctorReport` snapshot; an empty ``warnings`` tuple means no
    heuristic fired.
    """
    versions = {name: _package_version(name) for name in _CORE_PACKAGES}
    user_site = _user_site()
    pip_text = _pip_report()
    in_virtualenv = sys.prefix != sys.base_prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    user_site_on_path = bool(user_site and user_site in sys.path)
    wsl2 = _is_wsl2()
    nvidia_smi, nvidia_error = _nvidia_smi() if wsl2 else (None, None)
    backend, devices, jax_probe, jax_error = _jax_info()

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
    if wsl2 and backend in ("gpu", "cuda"):
        if not _version_at_least(versions["jaxlib"], "0.10.1"):
            warnings.append(
                f"WSL2 GPU is using jaxlib {versions['jaxlib']}. Upgrade JAX and "
                "jaxlib together to 0.10.1 or newer: this is the first verified "
                "release containing the upstream fixes for the spurious PJRT "
                "cache-hit warning and two-component Windows NVIDIA driver versions."
            )
        if nvidia_error is not None:
            warnings.append(nvidia_error)
    elif wsl2 and nvidia_smi is not None and backend not in (None, "gpu", "cuda"):
        warnings.append(
            f"nvidia-smi sees {nvidia_smi}, but JAX selected the {backend} backend."
        )

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
        wsl2=wsl2,
        nvidia_smi=nvidia_smi,
        jax_backend=backend,
        jax_default_device=_jax_default_device(),
        jax_devices=devices,
        jax_probe=jax_probe,
        warnings=tuple(warnings),
    )


def format_report(report: DoctorReport) -> str:
    """Render a :class:`DoctorReport` as the plain-text block the CLI prints.

    The layout is fixed: a header, the interpreter/platform/prefix block (with
    a ``Conda env`` line only when one is active), the aligned package-version
    table in ``_CORE_PACKAGES`` order, the JAX backend / default device /
    JIT-probe lines and the device list (``- none detected`` when empty), a
    ``WSL2 NVIDIA`` line only on WSL2, VMEX's three device-placement defaults
    (forward, implicit-gradient, mirror lanes), and one line for the
    persistent JAX compilation cache — its directory, used size and bound in
    GiB, flagged when it sits within 5 % of the bound, since a cache at its
    bound evicts what the next run needs.

    The last block is either ``Warnings:`` with one bullet per entry followed
    by the recommended clean-install commands, or, when ``report.warnings`` is
    empty, the single line ``Status: no obvious installation problems
    detected.``

    Parameters
    ----------
    report:
        Snapshot to render, normally straight from :func:`collect_report`.

    Returns
    -------
    The report as one newline-joined string, without a trailing newline.
    """
    lines = [
        "vmex installation doctor",
        "----------------------------",
        f"Python:      {report.python}",
        f"Executable:  {report.executable}",
        f"Prefix:      {report.prefix}",
        f"Base prefix: {report.base_prefix}",
        f"Platform:    {report.platform}",
        f"WSL2:        {'yes' if report.wsl2 else 'no'}",
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
            f"JAX JIT probe: {report.jax_probe or 'failed or unavailable'}",
            "JAX devices:",
        ]
    )
    if report.jax_devices:
        lines.extend(f"  - {device}" for device in report.jax_devices)
    else:
        lines.append("  - none detected")
    if report.wsl2:
        lines.append(f"WSL2 NVIDIA: {report.nvidia_smi or 'unavailable'}")
    lines.extend(
        [
            "VMEX forward default:  automatic work/mode-based CPU/GPU policy",
            "VMEX implicit default: CPU on accelerator hosts (explicit device= overrides)",
            "VMEX mirror default:   CPU for host-SciPy/JAX callback solves",
        ]
    )
    lines.append(_compilation_cache_line())
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
    """Run the installation doctor: collect, format, print.

    This is the entry point ``vmex --doctor`` dispatches to (see
    :mod:`vmex.core.cli`), which returns this value as the process exit code.

    Returns
    -------
    Always ``0``.  The exit status reports that the diagnostics ran, not that
    the environment is healthy — a report full of warnings still exits ``0``,
    so scripts must parse the printed text (or call :func:`collect_report` and
    inspect ``warnings``) rather than test the status.
    """
    print(format_report(collect_report()))
    return 0
