#!/usr/bin/env python3
"""Compare VMEC++ vs vmec_jax first-step diagnostics (axisymmetric cases)."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np

import vmec_jax as vj
from vmec_jax.diagnostics import summarize_many
from vmec_jax.solve import vmecpp_first_step_diagnostics


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.sqrt(np.mean(b * b))
    if denom == 0.0:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def _is_indata(path: Path) -> bool:
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            return stripped == "&INDATA"
    return False


def _find_indata2json(vmec_jax_root: Path) -> Path:
    env = os.getenv("VMECPP_INDATA2JSON")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    vmecpp_root = vmec_jax_root.parent / "vmecpp"
    candidates.extend(
        [
            vmecpp_root / "build311/_deps/indata2json-build/indata2json",
            vmecpp_root / "build2/_deps/indata2json-build/indata2json",
        ]
    )
    candidates.extend(vmecpp_root.glob("build*/_deps/indata2json-build/indata2json"))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "indata2json executable not found. Set VMECPP_INDATA2JSON or build vmecpp."
    )


def _ensure_vmecpp_input(input_path: Path, vmec_jax_root: Path) -> Path:
    if not _is_indata(input_path):
        return input_path
    indata2json = _find_indata2json(vmec_jax_root)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        local_input = tmpdir_path / input_path.name
        shutil.copyfile(input_path, local_input)
        subprocess.run(
            [str(indata2json), local_input.name],
            check=True,
            cwd=tmpdir_path,
        )
        json_name = input_path.name.replace("input.", "") + ".json"
        json_path = tmpdir_path / json_name
        if not json_path.is_file():
            raise RuntimeError(f"indata2json did not produce {json_path}")
        out_path = input_path.with_suffix(".json")
        shutil.copyfile(json_path, out_path)
        return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="circular_tokamak",
        choices=[
            "circular_tokamak",
            "shaped_tokamak_pressure",
            "vmecpp_solovev",
        ],
    )
    parser.add_argument("--step-size", type=float, default=None)
    args = parser.parse_args()

    try:
        import vmecpp
    except Exception as exc:
        raise SystemExit(f"vmecpp import failed: {exc}")

    vmec_jax_root = Path(__file__).resolve().parents[2]
    data_dir = vmec_jax_root / "examples" / "data"
    input_path = data_dir / f"input.{args.case}"

    _, indata0 = vj.load_input(input_path)
    ns_array = indata0.get("NS_ARRAY", 0)
    ns_override = ns_array[0] if isinstance(ns_array, list) and ns_array else None
    run = vj.run_fixed_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        ns_override=ns_override,
    )
    diag_jax = vmecpp_first_step_diagnostics(
        run.state,
        run.static,
        indata=run.indata,
        signgs=int(run.signgs),
        step_size=args.step_size,
        include_edge=True,
        zero_m1=True,
    )

    vmec_input_path = _ensure_vmecpp_input(input_path, vmec_jax_root)
    vmec_indata = vmecpp.cpp._vmecpp.VmecINDATA.from_file(vmec_input_path)
    diag_cpp = vmecpp.cpp._vmecpp.first_step_diagnostics(
        vmec_indata, max_threads=1, verbose=False
    )

    print(f"[axisym_step1_compare] case={args.case} input={input_path}")
    for key in ("fsqr", "fsqz", "fsql", "fsqr1", "fsqz1", "fsql1", "f_norm1"):
        j = float(diag_jax[key])
        c = float(diag_cpp[key])
        rel = abs(j - c) / max(abs(c), 1e-30)
        print(f"  {key}: jax={j:.6e} vmecpp={c:.6e} rel={rel:.3e}")

    for name in ("frcc", "fzsc", "flsc"):
        j = np.asarray(diag_jax[f"{name}_u"])
        c = np.asarray(diag_cpp[name])
        if j.shape != c.shape:
            print(f"  {name}: shape mismatch jax={j.shape} vmecpp={c.shape}")
            min_ns = min(j.shape[0], c.shape[0])
            j = j[:min_ns]
            c = c[:min_ns]
        print(f"  {name}: rel_rms={_rel_rms(j, c):.3e}")

    summarize_many(
        [
            ("jax_frcc_u", diag_jax["frcc_u"]),
            ("cpp_frcc", diag_cpp["frcc"]),
            ("jax_fzsc_u", diag_jax["fzsc_u"]),
            ("cpp_fzsc", diag_cpp["fzsc"]),
            ("jax_flsc_u", diag_jax["flsc_u"]),
            ("cpp_flsc", diag_cpp["flsc"]),
        ],
        indent="  ",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
