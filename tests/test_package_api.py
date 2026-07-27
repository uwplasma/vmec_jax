"""Package-surface tests: lazy exports, ``python -m vmex``, CLI helpers.

Covers the ``vmex/__init__.py`` lazy-attribute machinery (every
documented public name resolves; unknown names raise), the source-tree
version resolution, the ``__main__`` module entry point, and the small
pure CLI helpers (``case_from_input``, ``--booz-surfaces`` parsing, and
ESSOS coils-file loading for ``--coils``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

import vmex
from vmex.core import cli
from vmex.core.errors import VmecInputError
from vmex.core.mgrid import MgridField


# ---------------------------------------------------------------------------
# vmex/__init__.py
# ---------------------------------------------------------------------------


def test_version_matches_pyproject():
    src = vmex._source_tree_version()
    assert src is not None
    assert vmex.__version__ == src


def test_source_tree_version_parses_pyproject():
    # anchored to the PACKAGE (vmex/__init__.py), not this test file:
    # parents[1] of vmex/__init__.py is the repo root.
    pyproject = Path(vmex.__file__).resolve().parents[1] / "pyproject.toml"
    expected = None
    in_project = False
    for line in pyproject.read_text().splitlines():
        line = line.strip()
        if line == "[project]":
            in_project = True
        elif in_project and line.startswith("version"):
            expected = line.split("=", 1)[1].strip().strip('"')
            break
    assert vmex._source_tree_version() == expected


def test_every_lazy_export_resolves():
    for name in vmex._LAZY_ATTRS:
        value = getattr(vmex, name)
        assert value is not None, name
    # a second access hits the cached globals() entry
    assert getattr(vmex, "VmecInput") is vmex.VmecInput


def test_unknown_attribute_raises_and_dir_lists_exports():
    with pytest.raises(AttributeError):
        vmex.no_such_symbol  # noqa: B018
    listing = dir(vmex)
    for name in ("VmecInput", "solve", "solve_multigrid", "read_wout", "optimize"):
        assert name in listing
    assert set(vmex.__all__) >= set(vmex._LAZY_ATTRS)


def test_python_dash_m_entrypoint_exposes_cli_main():
    import vmex.__main__ as main_mod

    assert main_mod.main is cli.main


@pytest.mark.parametrize("argv", [["-m", "vmex", "--version"]])
def test_python_dash_m_version(argv):
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0
    assert vmex.__version__ in proc.stdout


# ---------------------------------------------------------------------------
# CLI pure helpers
# ---------------------------------------------------------------------------


def test_case_from_input_naming_conventions(tmp_path):
    assert cli.case_from_input(Path("input.solovev")) == "solovev"
    assert cli.case_from_input(Path("input_solovev")) == "solovev"
    assert cli.case_from_input(Path("solovev.json")) == "solovev"
    assert cli.case_from_input(Path("input.cth.json")) == "cth"


def test_parse_booz_surfaces():
    assert cli._parse_booz_surfaces(None) is None
    assert cli._parse_booz_surfaces("all") is None
    assert cli._parse_booz_surfaces("") is None
    assert cli._parse_booz_surfaces("0.25, 0.5 0.75") == [0.25, 0.5, 0.75]
    with pytest.raises(VmecInputError):
        cli._parse_booz_surfaces("0.25, banana")


def _coils_payload():
    rng = np.random.default_rng(0)
    return {
        "dofs_curves": rng.normal(size=(2, 3, 5)).tolist(),
        "dofs_currents": [1.0e5, 2.0e5],
        "n_segments": 32,
        "nfp": 4,
        "stellsym": True,
        "currents_scale": 2.0,
    }


def test_coils_mgrid_field_json_and_npz(tmp_path):
    # vmex is coil-agnostic: --coils loads ESSOS coils and tabulates them
    # into an in-memory mgrid, returning a plain MgridField.
    Coils = pytest.importorskip("essos.coils").Coils
    if not hasattr(Coils, "to_mgrid"):
        pytest.skip("ESSOS build lacks Coils.to_mgrid (coils->mgrid export)")
    payload = _coils_payload()

    jpath = tmp_path / "coils.json"
    jpath.write_text(json.dumps(payload))
    field_j = cli._coils_mgrid_field(jpath, nr=12, nphi=6, nz=12)
    assert isinstance(field_j, MgridField)

    npath = tmp_path / "coils.npz"
    np.savez(npath, **{k: np.asarray(v) for k, v in payload.items()})
    field_n = cli._coils_mgrid_field(npath, nr=12, nphi=6, nz=12)
    assert isinstance(field_n, MgridField)

    # A usable field: finite Biot-Savart at an interior cylindrical point.
    r = 0.5 * (float(field_n.rmin) + float(field_n.rmax))
    z = 0.5 * (float(field_n.zmin) + float(field_n.zmax))
    br, bp, bz = field_n.b_cyl(jnp.asarray(r), jnp.asarray(0.0), jnp.asarray(z))
    assert all(bool(np.isfinite(np.asarray(c))) for c in (br, bp, bz))


def test_coils_mgrid_field_rejects_malformed_file(tmp_path):
    pytest.importorskip("essos")
    bad = tmp_path / "coils.json"
    bad.write_text(json.dumps({"not_dofs": []}))
    with pytest.raises(VmecInputError):
        cli._coils_mgrid_field(bad)


# ---------------------------------------------------------------------------
# CLI dispatch: --doctor and the argparse usage-error contract
# ---------------------------------------------------------------------------


def test_cli_doctor_flag(capsys):
    assert cli.main(["--doctor"]) == 0
    assert "vmex installation doctor" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    [],                                # no target at all
    ["--plot"],                        # --plot with no path anywhere
    ["--doctor", "input.solovev"],     # --doctor takes no input
    ["--test", "input.solovev"],       # --test takes no input
    ["--test", "--plot", "x.nc"],      # --test and --plot are exclusive
])
def test_cli_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_cli_wout_and_boozmn_require_plot_or_booz(tmp_path):
    wout = tmp_path / "wout_x.nc"
    wout.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(wout)])
    assert exc.value.code == 2

    boozmn = tmp_path / "boozmn_x.nc"
    boozmn.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(boozmn)])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        cli.main(["--plot", str(wout), str(boozmn)])  # both target forms
    assert exc.value.code == 2


def test_cli_coils_flag_requires_free_boundary_deck(capsys, tmp_path):
    coils = tmp_path / "coils.json"
    coils.write_text(json.dumps(_coils_payload()))
    deck = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.solovev"
    rc = cli.main([str(deck), "--coils", str(coils), "--outdir", str(tmp_path)])
    assert rc != 0
    assert "LFREEB = T" in capsys.readouterr().out


def test_cli_direct_coils_requires_coils_argument(capsys, tmp_path):
    src = (Path(__file__).resolve().parents[1] / "examples" / "data"
           / "input.cth_like_free_bdy_lasym_small")
    text = src.read_text()
    assert "MGRID_FILE" in text
    deck = tmp_path / "input.direct_coils_case"
    deck.write_text(
        "\n".join(
            "  MGRID_FILE = 'DIRECT_COILS'" if "MGRID_FILE" in ln else ln
            for ln in text.splitlines()
        )
    )

    rc = cli.main([str(deck), "--outdir", str(tmp_path)])
    assert rc != 0
    assert "DIRECT_COILS" in capsys.readouterr().out

    rc = cli.main([str(deck), "--coils", str(tmp_path / "missing.json"),
                   "--outdir", str(tmp_path)])
    assert rc != 0
    assert "coils file not found" in capsys.readouterr().out


def test_cli_missing_input_is_zero_crash(capsys):
    rc = cli.main(["/nonexistent/input.notthere"])
    assert rc != 0
    out = capsys.readouterr().out
    assert "HINT" in out and "not found" in out


@pytest.mark.usefixtures("_module_jit_enabled")
def test_cli_bundled_test_verbose(tmp_path, capsys):
    """--test without --quiet prints the guided walkthrough (README quickstart)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("netCDF4")
    rc = cli.main(["--test", "--outdir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vmec bundled test" in out
    assert "Equivalent manual command:" in out
    assert "Bundled test complete" in out
    assert (tmp_path / "figures").is_dir()
