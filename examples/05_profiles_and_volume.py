"""Step-3: evaluate input profiles and compute a volume profile from sqrt(g).

This script extends the step-2 geometry kernel with:
  - VMEC-style 1D profiles from &INDATA (pressure, iota/current)
  - volume integrals from the Jacobian sqrt(g)

Outputs a compact `.npz` file with profiles and volume diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.diagnostics import print_summary, summarize_array
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.integrals import cumtrapz_s, dvds_from_sqrtg
from vmec_jax.profiles import eval_profiles
from vmec_jax.static import build_static


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="VMEC input file (INDATA)")
    p.add_argument("--out", type=str, default="profiles_step3.npz")
    p.add_argument("--verbose", action="store_true", help="Print extra debug information")
    args = p.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        raise SystemExit(f"error: input file not found: {inpath}")

    if has_jax():
        enable_x64(True)

    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    g = eval_geom(st0, static)

    prof = eval_profiles(indata, static.s)
    pressure = np.asarray(prof["pressure"])
    iota = np.asarray(prof.get("iota")) if "iota" in prof else None
    current = np.asarray(prof.get("current")) if "current" in prof else None

    dvds = np.asarray(dvds_from_sqrtg(g.sqrtg, static.grid.theta, static.grid.zeta, cfg.nfp))
    V = np.asarray(cumtrapz_s(dvds, static.s))

    print("\n==== vmec_jax step-3 profiles + volume ====")
    print(f"ns={cfg.ns} ntheta={cfg.ntheta} nzeta={cfg.nzeta} nfp={cfg.nfp}")

    print_summary(summarize_array("pressure(s) [Pa]", pressure), indent="")
    if iota is not None:
        print_summary(summarize_array("iota(s)", iota), indent="")
    if current is not None:
        print_summary(summarize_array("current I(s) (vmec units)", current), indent="")

    print_summary(summarize_array("dV/ds (per field period)", dvds), indent="")
    print_summary(summarize_array("V(s)  (per field period)", V), indent="")
    V_per_period = float(V[-1]) if V.size else float("nan")
    V_total = V_per_period * float(cfg.nfp)
    print(f"V_total (per field period) = {V_per_period:.8e}")
    print(f"V_total (full torus)      = {V_total:.8e}")

    if args.verbose:
        sqrtg = np.asarray(g.sqrtg)
        print("\n-- extra checks --")
        print_summary(summarize_array("sqrtg (all)", sqrtg), indent="")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        s=np.asarray(static.s),
        theta=np.asarray(static.grid.theta),
        zeta=np.asarray(static.grid.zeta),
        nfp=np.asarray(cfg.nfp),
        pressure=pressure,
        dvds=dvds,
        V=V,
        ncurr=np.asarray(prof.get("ncurr", -1)),
        pmass_type=np.asarray(str(indata.get("PMASS_TYPE", "power_series"))),
        piota_type=np.asarray(str(indata.get("PIOTA_TYPE", "power_series"))),
        pcurr_type=np.asarray(str(indata.get("PCURR_TYPE", "power_series"))),
    )
    if iota is not None:
        save["iota"] = iota
    if current is not None:
        save["current"] = current

    np.savez(out, **save)
    print(f"saving: {out}")


if __name__ == "__main__":
    main()

