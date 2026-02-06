"""Compatibility wrapper for the n3are VMEC-vs-vmec_jax comparison script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_main():
    root = Path(__file__).resolve().parents[2]
    target = root / "examples" / "validation" / "n3are_vmec_vs_vmecjax.py"
    spec = importlib.util.spec_from_file_location("n3are_vmec_vs_vmecjax", target)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Could not load module from {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


if __name__ == "__main__":
    _load_main()()
