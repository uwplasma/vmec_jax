"""High-level helpers for VMEC driver scripts.

These functions provide a thin, convenient layer over the core modules so
simple scripts can be written with minimal boilerplate, while still allowing
power users to drop down to lower-level APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .boundary import boundary_from_indata
from .config import VMECConfig, load_config
from .energy import flux_profiles_from_indata
from .field import signgs_from_sqrtg
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .profiles import eval_profiles
from .solve import solve_fixed_boundary_gd, solve_fixed_boundary_lbfgs
from .static import VMECStatic, build_static
from .wout import WoutData, read_wout, state_from_wout


@dataclass(frozen=True)
class ExampleData:
    input_path: Path
    wout_path: Optional[Path]
    cfg: VMECConfig
    indata: any
    static: VMECStatic
    wout: Optional[WoutData]
    state: Optional[any]


@dataclass(frozen=True)
class FixedBoundaryRun:
    """Container returned by ``run_fixed_boundary``."""

    cfg: VMECConfig
    indata: any
    static: VMECStatic
    state: any
    result: any | None
    flux: any
    profiles: dict
    signgs: int


def example_paths(case: str, *, root: str | Path | None = None) -> tuple[Path, Optional[Path]]:
    """Return (input_path, wout_path) for a bundled example case."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    data_dir = root / "examples" / "data"
    input_path = data_dir / f"input.{case}"
    wout_path = data_dir / f"wout_{case}_reference.nc"
    if not wout_path.exists():
        wout_path = data_dir / f"wout_{case}.nc"
    if not wout_path.exists():
        wout_path = None
    return input_path, wout_path


def load_example(
    case: str,
    *,
    root: str | Path | None = None,
    with_wout: bool = True,
    grid=None,
) -> ExampleData:
    """Load a bundled example case (config + static + optional wout/state)."""
    input_path, wout_path = example_paths(case, root=root)
    cfg, indata = load_config(str(input_path))
    static = build_static(cfg, grid=grid)
    if with_wout and wout_path is not None:
        wout = read_wout(wout_path)
        state = state_from_wout(wout)
    else:
        wout = None
        state = None
    return ExampleData(
        input_path=input_path,
        wout_path=wout_path,
        cfg=cfg,
        indata=indata,
        static=static,
        wout=wout,
        state=state,
    )


def load_input(path: str | Path):
    """Convenience wrapper around `load_config`."""
    return load_config(str(path))


def load_wout(path: str | Path) -> WoutData:
    """Convenience wrapper around `read_wout`."""
    return read_wout(path)


def save_npz(path: str | Path, **arrays) -> Path:
    """Save arrays into a NumPy `.npz` file and return the path."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def run_fixed_boundary(
    input_path: str | Path,
    *,
    solver: str = "gd",
    max_iter: int = 20,
    step_size: float = 5e-3,
    history_size: int = 10,
    use_initial_guess: bool = False,
    grid=None,
):
    """Run a fixed-boundary vmec_jax solve with minimal boilerplate.

    Parameters
    ----------
    solver:
        ``"gd"`` (gradient descent) or ``"lbfgs"``.
    use_initial_guess:
        If True, skip the solve and return the initialized state.
    """
    cfg, indata = load_config(str(input_path))
    static = build_static(cfg, grid=grid)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    g0 = eval_geom(st0, static)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    prof = eval_profiles(indata, static.s)
    pressure = prof.get("pressure", np.zeros_like(np.asarray(static.s)))
    gamma = indata.get_float("GAMMA", 0.0)

    if use_initial_guess:
        return FixedBoundaryRun(
            cfg=cfg,
            indata=indata,
            static=static,
            state=st0,
            result=None,
            flux=flux,
            profiles=prof,
            signgs=signgs,
        )

    solver = solver.lower()
    if solver == "gd":
        res = solve_fixed_boundary_gd(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(max_iter),
            step_size=float(step_size),
            jacobian_penalty=1e3,
            jit_grad=True,
            verbose=False,
        )
    elif solver == "lbfgs":
        res = solve_fixed_boundary_lbfgs(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(max_iter),
            step_size=float(step_size),
            history_size=int(history_size),
            jit_grad=True,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown solver: {solver!r} (expected 'gd' or 'lbfgs')")

    return FixedBoundaryRun(
        cfg=cfg,
        indata=indata,
        static=static,
        state=res.state,
        result=res,
        flux=flux,
        profiles=prof,
        signgs=signgs,
    )
