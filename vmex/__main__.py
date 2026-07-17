"""Module entry point for `python -m vmex`."""

import os as _os

# Suppress noisy C++ warnings from XLA/PjRt before any JAX import.
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
_os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
_os.environ.setdefault("GLOG_minloglevel", "2")

from .core.cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
