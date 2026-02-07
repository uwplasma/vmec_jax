from __future__ import annotations

import numpy as np
from pathlib import Path

from vmec_jax.field import chips_from_wout_chipf, full_mesh_from_half_mesh_avg
from vmec_jax.wout import read_wout


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sqrt(np.mean((a - b) ** 2))
    den = np.sqrt(np.mean(b**2))
    return float(num / den) if den > 0 else float("inf")


def test_chips_from_wout_chipf_detects_half_mesh_vmec2000_style():
    repo_root = Path(__file__).resolve().parents[1]
    wout = read_wout(repo_root / "examples" / "data" / "wout_n3are_R7.75B5.7_lowres.nc")
    chipf = np.asarray(wout.chipf)
    phipf = np.asarray(wout.phipf)
    iotaf = np.asarray(wout.iotaf)
    iotas = np.asarray(wout.iotas)

    chips = np.asarray(
        chips_from_wout_chipf(chipf=chipf, phipf=phipf, iotaf=iotaf, iotas=iotas, assume_half_if_unknown=True)
    )
    chips_half = np.asarray(full_mesh_from_half_mesh_avg(chipf))

    err_to_half = _rel_rms(chips, chips_half)
    err_to_full = _rel_rms(chips, chipf)
    assert err_to_half < err_to_full


def test_chips_from_wout_chipf_detects_full_mesh_vmecpp_style():
    # Synthetic case: chipf follows full-mesh iotas*phipf much closer than iotaf*phipf.
    ns = 8
    s = np.linspace(0.0, 1.0, ns)
    phipf = 1.0 + 0.1 * s
    iotas = 0.3 + 0.2 * s
    iotaf = 0.3 + 0.2 * (0.5 * (s + np.roll(s, 1)))
    iotaf[0] = iotaf[1]
    chipf_full = iotas * phipf

    chips = np.asarray(
        chips_from_wout_chipf(
            chipf=chipf_full,
            phipf=phipf,
            iotaf=iotaf,
            iotas=iotas,
            assume_half_if_unknown=True,
        )
    )
    assert np.allclose(chips, chipf_full, rtol=0, atol=1e-12)


def test_chips_from_wout_chipf_prefers_full_when_slightly_better():
    # Full-mesh should be selected whenever it is the closer interpretation,
    # even if the margin over half-mesh is modest.
    ns = 16
    s = np.linspace(0.0, 1.0, ns)
    phipf = 1.2 + 0.05 * s
    iotas = 0.28 + 0.15 * s
    iotaf = iotas + 0.03 * (1.0 - s)  # similar, but systematically offset
    chipf = (iotas * phipf) + 0.002 * np.sin(2.0 * np.pi * s)

    chips = np.asarray(
        chips_from_wout_chipf(
            chipf=chipf,
            phipf=phipf,
            iotaf=iotaf,
            iotas=iotas,
            assume_half_if_unknown=True,
        )
    )
    assert _rel_rms(chips, chipf) < _rel_rms(chips, full_mesh_from_half_mesh_avg(chipf))
