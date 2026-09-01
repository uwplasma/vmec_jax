from __future__ import annotations

from importlib.metadata import version as package_version

from packaging.requirements import Requirement
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_discovery_only_packages_vmex_namespace() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_find = data["tool"]["setuptools"]["packages"]["find"]

    assert package_find["where"] == ["."]
    # "vmex*" is the package; "vmec_jax" is the one-release deprecation shim.
    assert package_find["include"] == ["vmex*", "vmec_jax"]
    for pattern in ("tests*", "docs*", "examples*", "tools*", "validation*", "results*", "build*", "dist*"):
        assert pattern in package_find["exclude"]


def test_package_exposes_installed_version() -> None:
    import vmex

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert vmex.__version__ == data["project"]["version"]
    if not Path(vmex.__file__).resolve().is_relative_to(ROOT):
        assert vmex.__version__ == package_version("vmex")


def test_project_metadata_has_public_package_links() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data["project"]

    assert "stellarator" in project["keywords"]
    assert "Topic :: Scientific/Engineering :: Physics" in project["classifiers"]
    assert project["urls"]["Documentation"] == "https://vmex.readthedocs.io/en/latest/"
    assert project["urls"]["Repository"] == "https://github.com/uwplasma/vmex"
    assert project["urls"]["Changelog"] == "https://github.com/uwplasma/vmex/releases"


def test_project_exposes_vmec_console_aliases() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]

    # VMEX CLI: primary `vmex` + the legacy `vmec` alias.
    assert scripts["vmex"] == "vmex.core.cli:main"
    assert scripts["vmec"] == "vmex.core.cli:main"
    assert set(scripts) == {"vmex", "vmec"}


def test_plain_install_includes_plotting_and_qi_dependencies() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project_dependencies = set(data["project"]["dependencies"])
    # Compare parsed requirement NAMES, not raw strings: a version floor on
    # any dependency (booz_xform_jax>=0.1.1 broke the old string match) must
    # not trip the plain-install guard.
    dependency_names = {
        Requirement(dep).name for dep in project_dependencies
    }
    optional_dependencies = data.get("project", {}).get("optional-dependencies", {})

    assert "matplotlib" in dependency_names
    assert "booz_xform_jax" in dependency_names
    assert "packaging" in dependency_names
    assert "numpy" in dependency_names
    assert "solvax>=0.20.0" in project_dependencies
    assert "plots" not in optional_dependencies
    assert "plot" not in optional_dependencies
    assert "qi" not in optional_dependencies
    assert "booz" not in optional_dependencies


def test_build_system_declares_setuptools_license_validation_dependency() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    build_requires = set(data["build-system"]["requires"])

    assert "setuptools" in build_requires
    assert "packaging" in build_requires
