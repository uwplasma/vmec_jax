"""High-level configuration extracted from VMEC input (&INDATA).

For step-0 we only extract what we need for geometry/basis:
- mpol, ntor, ns, nfp, lasym
- ntheta/nzeta defaults (VMEC conventions)

Later steps will extend this config to include profiles, iteration controls, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .namelist import InData, read_indata
from .modes import default_grid_sizes


@dataclass(frozen=True)
class VMECConfig:
    mpol: int
    ntor: int
    ns: int
    nfp: int
    lasym: bool
    lthreed: bool
    lconm1: bool
    ntheta: int
    nzeta: int


def config_from_indata(indata: InData) -> VMECConfig:
    mpol = indata.get_int("MPOL", 6)
    ntor = indata.get_int("NTOR", 0)
    # VMEC commonly uses NS_ARRAY = [coarse, ..., fine]. For setup we want the *finest*.
    ns_array = indata.get("NS_ARRAY", 0)
    if isinstance(ns_array, list) and ns_array:
        ns = int(ns_array[-1])
    else:
        ns = indata.get_int("NS_ARRAY", 0)
    if ns == 0:
        ns = indata.get_int("NS", 31)  # fallback (some inputs use NS instead)
    nfp = indata.get_int("NFP", 1)
    lasym = indata.get_bool("LASYM", False)
    # VMEC convention: lthreed is derived from toroidal modes (ntor>0).
    lthreed = bool(ntor > 0)
    lconm1 = indata.get_bool("LCONM1", True)
    ntheta_in = indata.get_int("NTHETA", 0)
    nzeta_in = indata.get_int("NZETA", 0)
    ntheta, nzeta = default_grid_sizes(mpol=mpol, ntor=ntor, ntheta=ntheta_in, nzeta=nzeta_in)
    return VMECConfig(
        mpol=mpol,
        ntor=ntor,
        ns=ns,
        nfp=nfp,
        lasym=lasym,
        lthreed=lthreed,
        lconm1=lconm1,
        ntheta=ntheta,
        nzeta=nzeta,
    )


def load_config(path: str | Path) -> tuple[VMECConfig, InData]:
    indata = read_indata(path)
    cfg = config_from_indata(indata)
    return cfg, indata
