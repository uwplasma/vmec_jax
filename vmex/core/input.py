"""VMEC input handling: the ``&INDATA`` namelist and VMEC++-style JSON.

VMEC2000 counterparts: ``LIBSTELL/Sources/Modules/vmec_input.f``
(``read_indata_namelist``: variable set and defaults) and ``readin.f``
(post-read normalizations).  The JSON schema follows VMEC++
(``vmecpp.VmecInput``): identical key names, boundary coefficients as sparse
``{"m": int, "n": int, "value": float}`` lists, dense axis arrays, and
``adiabatic_index`` accepted as an alias for ``gamma``.

:class:`VmecInput` is a frozen dataclass holding the full INDATA content this
code base consumes, with VMEC2000 defaults.  Parsing is host-side NumPy code
(nothing here needs JAX).

Normalizations applied on construction (all from VMEC2000):

* ``read_indata_namelist``: ``raxis_s[0] = 0`` and ``zaxis_s[0] = 0``; the
  obsolete ``RAXIS``/``ZAXIS`` arrays override ``RAXIS_CC``/``ZAXIS_CS`` where
  nonzero; ``niter_array`` falls back to ``NITER`` when absent.
* ``readin.f``: ``lfreeb`` is forced ``False`` when ``mgrid_file == 'NONE'``;
  ``nvacskip <= 0`` falls back to ``nfp``.
* Boundary coefficients outside ``|n| <= ntor``, ``0 <= m < mpol`` are
  dropped (VMEC2000 reads them into oversized arrays but never uses them).

Index conventions: ``rbc/zbs/rbs/zbc`` are dense 2D arrays of shape
``(2*ntor + 1, mpol)`` indexed ``[n + ntor, m]``, i.e. ``rbc[n + ntor, m]``
is the INDATA coefficient ``RBC(n, m)``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

__all__ = ["VmecInput"]

Scalar = Union[str, bool, int, float]

# ---------------------------------------------------------------------------
# Tolerant Fortran-namelist tokenizer (targeted at VMEC &INDATA files)
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(r"(?P<key>[A-Za-z_]\w*(?:\([^\)]*\))?)\s*=", re.MULTILINE)
_REPEAT_RE = re.compile(r"^(?P<count>\d+)\*(?P<value>.+)$")
_BOOL_TRUE = {"T", ".T.", ".TRUE.", "TRUE"}
_BOOL_FALSE = {"F", ".F.", ".FALSE.", "FALSE"}


def _strip_fortran_comments(line: str) -> str:
    """Remove ``!`` comments, respecting single-quoted strings."""
    out: List[str] = []
    in_quote = False
    for ch in line:
        if ch == "'":
            in_quote = not in_quote
        elif ch == "!" and not in_quote:
            break
        out.append(ch)
    return "".join(out)


def _tokenize_values(chunk: str) -> List[str]:
    """Split a value chunk into tokens, keeping quoted strings intact."""
    tokens: List[str] = []
    buf: List[str] = []
    in_quote = False
    for ch in chunk.strip():
        if ch == "'":
            in_quote = not in_quote
            buf.append(ch)
        elif not in_quote and ch in ", \t\r\n":
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    # Expand Fortran repeat syntax like ``11*0.0``.
    out: List[str] = []
    for tok in tokens:
        m = _REPEAT_RE.match(tok)
        if m and int(m.group("count")) > 0 and m.group("value").strip():
            out.extend([m.group("value").strip()] * int(m.group("count")))
        else:
            out.append(tok)
    return out


def _parse_scalar(tok: str) -> Scalar:
    """Parse one token: quoted string, logical, integer, or (D-exponent) float."""
    tok = tok.strip()
    for quote in ("'", '"'):
        if len(tok) >= 2 and tok[0] == quote and tok[-1] == quote:
            return tok[1:-1].strip()
    up = tok.upper()
    if up in _BOOL_TRUE:
        return True
    if up in _BOOL_FALSE:
        return False
    if re.fullmatch(r"[+-]?\d+", tok):
        return int(tok)
    try:
        return float(tok.replace("D", "E").replace("d", "E"))
    except ValueError:
        return tok


def _parse_key(key: str) -> Tuple[str, Tuple[int, ...] | None]:
    """Split ``KEY`` or ``KEY(i,j)`` into (name, indices); ``KEY(:)`` -> plain."""
    key = key.strip()
    if "(" not in key:
        return key.upper(), None
    base, rest = key.split("(", 1)
    rest = rest.rstrip(")")
    if ":" in rest:
        return base.upper(), None
    return base.upper(), tuple(int(x) for x in rest.split(",") if x.strip())


def _read_indata_text(text: str) -> tuple[Dict[str, List[Scalar]], Dict[str, Dict[Tuple[int, ...], Scalar]]]:
    """Parse the ``&INDATA`` block of ``text`` into scalar and indexed maps.

    Returns ``(scalars, indexed)`` where ``scalars`` maps upper-case names to
    token lists and ``indexed`` maps names like ``RBC`` to ``{(n, m): value}``.
    """
    m_start = re.search(r"&\s*INDATA", text, flags=re.IGNORECASE)
    if not m_start:
        raise ValueError("no &INDATA namelist found")
    m_end = re.search(r"\n\s*/\s*\n|\n\s*/\s*$", text[m_start.end():], flags=re.MULTILINE)
    if not m_end:
        raise ValueError("no terminating '/' for &INDATA")
    block = text[m_start.end(): m_start.end() + m_end.start()]
    cleaned = "\n".join(_strip_fortran_comments(ln) for ln in block.splitlines())

    scalars: Dict[str, List[Scalar]] = {}
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]] = {}
    matches = list(_ASSIGN_RE.finditer(cleaned))
    for i, m in enumerate(matches):
        name, idx = _parse_key(m.group("key"))
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        chunk = re.sub(r",\s*$", "", cleaned[m.end(): val_end].strip())
        values = [_parse_scalar(t) for t in _tokenize_values(chunk)]
        if not values:
            continue
        if idx is None:
            scalars[name] = values
        else:
            indexed.setdefault(name, {})[idx] = values[0]
    return scalars, indexed


# ---------------------------------------------------------------------------
# VmecInput
# ---------------------------------------------------------------------------


def _float_array(values, dtype=np.float64) -> np.ndarray:
    if values is None:
        return np.zeros((0,), dtype=dtype)
    return np.atleast_1d(np.asarray(values, dtype=dtype)).ravel()


def _dense_min_length(values, n: int) -> np.ndarray:
    """Dense float array zero-padded on the right to length >= ``n``."""
    arr = _float_array(values)
    if arr.size >= n:
        return arr
    return np.pad(arr, (0, n - arr.size))


def _fixed_length(values, n: int, fill: float = 0.0) -> np.ndarray:
    """Dense float array of exact length ``n`` (truncate or pad with ``fill``)."""
    arr = _float_array(values)
    out = np.full((n,), float(fill), dtype=np.float64)
    k = min(arr.size, n)
    out[:k] = arr[:k]
    return out


def _trim_aux(aux_s, aux_f) -> tuple[np.ndarray, np.ndarray]:
    """Trim spline knots to the strictly increasing leading segment.

    VMEC2000 fills unset ``*_AUX_S`` entries with -1 and locates the active
    knot count via ``minloc(aux_s(2:))`` (profile_functions.f); for the
    increasing knot vectors used in practice this is equivalent to cutting at
    the first non-increasing entry.  Both arrays are trimmed to the common
    valid length.
    """
    s = _float_array(aux_s)
    f = _float_array(aux_f)
    n = min(s.size, f.size)
    if n == 0:
        return np.zeros((0,)), np.zeros((0,))
    n_valid = n
    for idx in range(1, n):
        if s[idx] <= s[idx - 1]:
            n_valid = idx
            break
    return s[:n_valid].copy(), f[:n_valid].copy()


@dataclass(frozen=True, eq=False)
class VmecInput:
    """Full ``&INDATA`` content with VMEC2000 semantics and defaults.

    Defaults are the initializations in ``read_indata_namelist``
    (``vmec_input.f``), after the ``readin.f`` normalizations documented in
    the module docstring.  Array fields are NumPy arrays; ``None`` defaults
    are resolved in ``__post_init__`` (they depend on ``mpol``/``ntor``).
    """

    # -- symmetry / resolution (readin.f) --
    lasym: bool = False          #: non-stellarator-symmetric mode
    nfp: int = 1                 #: number of field periods
    mpol: int = 6                #: poloidal modes m = 0..mpol-1
    ntor: int = 0                #: toroidal modes n = -ntor..ntor
    ntheta: int = 0              #: poloidal grid points (0 -> VMEC default)
    nzeta: int = 0               #: toroidal grid points (0 -> VMEC default)

    # -- multigrid ladder / stepping (runvmec.f, evolve.f) --
    ns_array: Any = None         #: radial surfaces per stage (default [31])
    ftol_array: Any = None       #: force tolerance per stage (default [1e-10])
    niter_array: Any = None      #: iteration cap per stage (default [100] = NITER)
    delt: float = 1.0            #: initial time step
    tcon0: float = 1.0           #: constraint-force multiplier (bcovar.f)
    aphi: Any = None             #: radial-flux remap polynomial (default [1,0,...], len 20)
    phiedge: float = 1.0         #: total enclosed toroidal flux [Wb]
    nstep: int = 10              #: iterations between progress prints

    # -- pressure profile (pmass; Pa before mu0 conversion) --
    pmass_type: str = "power_series"
    am: Any = None               #: pmass coefficients (dense, len >= 21)
    am_aux_s: Any = None         #: pmass spline knots s
    am_aux_f: Any = None         #: pmass spline values
    pres_scale: float = 1.0      #: pressure scale factor [Pa]
    gamma: float = 0.0           #: adiabatic index (JSON: also 'adiabatic_index')
    spres_ped: float = 1.0       #: pressure pedestal s (profil1d.f clamp)

    # -- current / iota profiles (pcurr / piota) --
    ncurr: int = 0               #: 0: prescribed iota, 1: prescribed current
    pcurr_type: str = "power_series"
    ac: Any = None               #: pcurr coefficients (dense, len >= 21)
    ac_aux_s: Any = None
    ac_aux_f: Any = None
    curtor: float = 0.0          #: total toroidal current [A]
    piota_type: str = "power_series"
    ai: Any = None               #: piota coefficients (dense, len >= 21)
    ai_aux_s: Any = None
    ai_aux_f: Any = None
    bloat: float = 1.0           #: profile-argument expansion factor

    # -- axis initial guess (n = 0..ntor) --
    raxis_c: Any = None          #: R axis cos coefficients (INDATA RAXIS_CC)
    zaxis_s: Any = None          #: Z axis sin coefficients (INDATA ZAXIS_CS)
    raxis_s: Any = None          #: R axis sin coefficients (lasym; RAXIS_CS)
    zaxis_c: Any = None          #: Z axis cos coefficients (lasym; ZAXIS_CC)

    # -- boundary coefficients, dense [n + ntor, m] of shape (2*ntor+1, mpol) --
    rbc: Any = None              #: R boundary cos(m u - n nfp v)
    zbs: Any = None              #: Z boundary sin(m u - n nfp v)
    rbs: Any = None              #: R boundary sin (lasym)
    zbc: Any = None              #: Z boundary cos (lasym)

    # -- free boundary (readin.f) --
    lfreeb: bool = True          #: forced False when mgrid_file == 'NONE'
    mgrid_file: str = "NONE"
    extcur: Any = None           #: external coil-group currents [A]
    nvacskip: int = 1            #: vacuum-solve cadence (<= 0 -> nfp)

    # -- boundary spectral filtering / preconditioner --
    mfilter_fbdy: int = -1
    nfilter_fbdy: int = -1
    precon_type: str = "NONE"
    prec2d_threshold: float = 1e-30

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "lasym", bool(self.lasym))
        for name in ("nfp", "mpol", "ntor", "ntheta", "nzeta", "ncurr", "nstep",
                     "nvacskip", "mfilter_fbdy", "nfilter_fbdy"):
            set_(self, name, int(getattr(self, name)))
        for name in ("delt", "tcon0", "phiedge", "pres_scale", "gamma", "spres_ped",
                     "curtor", "bloat", "prec2d_threshold"):
            set_(self, name, float(getattr(self, name)))
        for name in ("pmass_type", "pcurr_type", "piota_type"):
            set_(self, name, str(getattr(self, name)).strip().lower())
        set_(self, "precon_type", str(self.precon_type).strip())
        set_(self, "mgrid_file", str(self.mgrid_file).strip())

        # Multigrid ladder: ns_array trimmed to its positive prefix; ftol and
        # niter arrays resized to the same number of stages (missing trailing
        # entries repeat the last given value).
        ns = np.atleast_1d(np.asarray(
            [31] if self.ns_array is None else self.ns_array, dtype=np.int64)).ravel()
        n_stages = int(np.argmax(ns <= 0)) if np.any(ns <= 0) else ns.size
        n_stages = max(n_stages, 1)
        set_(self, "ns_array", ns[:n_stages].copy())
        ftol = _float_array([1e-10] if self.ftol_array is None else self.ftol_array)
        niter = np.atleast_1d(np.asarray(
            [100] if self.niter_array is None else self.niter_array, dtype=np.int64)).ravel()
        set_(self, "ftol_array", _fixed_length(ftol, n_stages, fill=float(ftol[-1]))
             if ftol.size else np.full((n_stages,), 1e-10))
        niter_full = np.full((n_stages,), int(niter[-1]) if niter.size else 100, dtype=np.int64)
        niter_full[: min(niter.size, n_stages)] = niter[: min(niter.size, n_stages)]
        set_(self, "niter_array", niter_full)

        # aphi: length 20, default [1, 0, ...] (vmec_input.f: aphi=0; aphi(1)=1).
        if self.aphi is None:
            aphi = np.zeros((20,)); aphi[0] = 1.0
        else:
            aphi = _fixed_length(self.aphi, 20)
        set_(self, "aphi", aphi)

        # Profile coefficient arrays: dense, at least the VMEC (0:20) extent.
        for name in ("am", "ac", "ai"):
            set_(self, name, _dense_min_length(getattr(self, name), 21))
        for pre in ("am", "ac", "ai"):
            s_arr, f_arr = _trim_aux(getattr(self, f"{pre}_aux_s"), getattr(self, f"{pre}_aux_f"))
            set_(self, f"{pre}_aux_s", s_arr)
            set_(self, f"{pre}_aux_f", f_arr)

        # Axis arrays: dense length ntor+1; read_indata_namelist zeroes the
        # n=0 sine coefficients (raxis_cs(0) = 0; zaxis_cs(0) = 0).
        for name in ("raxis_c", "zaxis_s", "raxis_s", "zaxis_c"):
            set_(self, name, _fixed_length(getattr(self, name), self.ntor + 1))
        self.raxis_s[0] = 0.0
        self.zaxis_s[0] = 0.0

        # Boundary coefficients: dense (2*ntor+1, mpol) indexed [n+ntor, m].
        shape = (2 * self.ntor + 1, self.mpol)
        for name in ("rbc", "zbs", "rbs", "zbc"):
            value = getattr(self, name)
            arr = np.zeros(shape) if value is None else np.asarray(value, dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
            set_(self, name, arr.copy())

        set_(self, "extcur", _float_array(self.extcur))
        # readin.f: IF (lfreeb .and. mgrid_file == 'NONE') lfreeb = .false.
        set_(self, "lfreeb", bool(self.lfreeb) and self.mgrid_file.upper() != "NONE")
        # readin.f: IF (nvacskip <= 0) nvacskip = nfp
        if self.nvacskip <= 0:
            set_(self, "nvacskip", self.nfp)

    # -- equality -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Field-wise equality with exact array comparison."""
        if not isinstance(other, VmecInput):
            return NotImplemented
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not (np.shape(a) == np.shape(b) and np.array_equal(a, b)):
                    return False
            elif a != b:
                return False
        return True

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "VmecInput":
        """Read a VMEC input file, auto-detecting INDATA vs JSON format.

        Files whose first non-whitespace character is ``{`` (or with a
        ``.json`` suffix) are parsed as VMEC++-style JSON; everything else as
        a classic ``&INDATA`` Fortran namelist (VMEC2000 ``readin.f``).
        """
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() == ".json" or text.lstrip()[:1] == "{":
            return cls.from_json_text(text)
        return cls.from_indata_text(text)

    @classmethod
    def from_indata_text(cls, text: str) -> "VmecInput":
        """Build from ``&INDATA`` namelist text (VMEC2000 read_indata_namelist)."""
        scalars, indexed = _read_indata_text(text)

        def get(name: str, default=None):
            values = scalars.get(name)
            if not values:
                return default
            return values[0] if len(values) == 1 else values

        def get_list(name: str) -> list | None:
            values = scalars.get(name)
            if values is None:
                return None
            return list(values)

        mpol = int(get("MPOL", 6))
        ntor = int(get("NTOR", 0))

        # ns_array: NS_ARRAY, or legacy NS, or the VMEC default 31.
        ns_array = get_list("NS_ARRAY")
        if ns_array is None:
            ns_array = [int(get("NS", 31))]
        # ftol_array: FTOL_ARRAY, or scalar FTOL (vmec_input.f: ftol_array(1)=ftol).
        ftol_array = get_list("FTOL_ARRAY")
        if ftol_array is None:
            ftol_array = [float(get("FTOL", 1e-10))]
        # niter_array falls back to NITER when absent (read_indata_namelist).
        niter_array = get_list("NITER_ARRAY")
        if niter_array is None:
            niter_array = [int(get("NITER", 100))]

        def axis(name: str, legacy: str | None = None) -> np.ndarray:
            out = _fixed_length(get_list(name) or [], ntor + 1)
            for idx, value in indexed.get(name, {}).items():
                if len(idx) == 1 and 0 <= idx[0] <= ntor:
                    out[idx[0]] = float(value)
            if legacy is not None:
                # Backwards compatibility (read_indata_namelist):
                # WHERE (raxis /= 0) raxis_cc = raxis (idem zaxis -> zaxis_cs).
                old = _fixed_length(get_list(legacy) or [], ntor + 1)
                out = np.where(old != 0.0, old, out)
            return out

        def boundary(name: str) -> np.ndarray:
            grid = np.zeros((2 * ntor + 1, mpol))
            for idx, value in indexed.get(name, {}).items():
                if len(idx) != 2:
                    continue
                n, m = idx
                if -ntor <= n <= ntor and 0 <= m < mpol:
                    grid[n + ntor, m] = float(value)
            return grid

        extcur_indexed = indexed.get("EXTCUR", {})
        if extcur_indexed:
            size = max(idx[0] for idx in extcur_indexed if len(idx) == 1)
            extcur = np.zeros((size,))
            for idx, value in extcur_indexed.items():
                if len(idx) == 1 and idx[0] >= 1:
                    extcur[idx[0] - 1] = float(value)
        else:
            extcur = _float_array(get_list("EXTCUR") or [])

        return cls(
            lasym=bool(get("LASYM", False)),
            nfp=int(get("NFP", 1)),
            mpol=mpol,
            ntor=ntor,
            ntheta=int(get("NTHETA", 0)),
            nzeta=int(get("NZETA", 0)),
            ns_array=ns_array,
            ftol_array=ftol_array,
            niter_array=niter_array,
            delt=float(get("DELT", 1.0)),
            tcon0=float(get("TCON0", 1.0)),
            aphi=get_list("APHI"),
            phiedge=float(get("PHIEDGE", 1.0)),
            nstep=int(get("NSTEP", 10)),
            pmass_type=str(get("PMASS_TYPE", "power_series")),
            am=get_list("AM") or [],
            am_aux_s=get_list("AM_AUX_S") or [],
            am_aux_f=get_list("AM_AUX_F") or [],
            pres_scale=float(get("PRES_SCALE", 1.0)),
            gamma=float(get("GAMMA", 0.0)),
            spres_ped=float(get("SPRES_PED", 1.0)),
            ncurr=int(get("NCURR", 0)),
            pcurr_type=str(get("PCURR_TYPE", "power_series")),
            ac=get_list("AC") or [],
            ac_aux_s=get_list("AC_AUX_S") or [],
            ac_aux_f=get_list("AC_AUX_F") or [],
            curtor=float(get("CURTOR", 0.0)),
            piota_type=str(get("PIOTA_TYPE", "power_series")),
            ai=get_list("AI") or [],
            ai_aux_s=get_list("AI_AUX_S") or [],
            ai_aux_f=get_list("AI_AUX_F") or [],
            bloat=float(get("BLOAT", 1.0)),
            raxis_c=axis("RAXIS_CC", legacy="RAXIS"),
            zaxis_s=axis("ZAXIS_CS", legacy="ZAXIS"),
            raxis_s=axis("RAXIS_CS"),
            zaxis_c=axis("ZAXIS_CC"),
            rbc=boundary("RBC"),
            zbs=boundary("ZBS"),
            rbs=boundary("RBS"),
            zbc=boundary("ZBC"),
            lfreeb=bool(get("LFREEB", True)),
            mgrid_file=str(get("MGRID_FILE", "NONE")),
            extcur=extcur,
            nvacskip=int(get("NVACSKIP", 1)),
            mfilter_fbdy=int(get("MFILTER_FBDY", -1)),
            nfilter_fbdy=int(get("NFILTER_FBDY", -1)),
            precon_type=str(get("PRECON_TYPE", "NONE")),
            prec2d_threshold=float(get("PREC2D_THRESHOLD", 1e-30)),
        )

    @classmethod
    def from_json_text(cls, text: str) -> "VmecInput":
        """Build from VMEC++-style JSON text (plan Appendix C / vmecpp.VmecInput).

        Same key names as the dataclass fields; ``adiabatic_index`` is
        accepted as an alias for ``gamma``; ``rbc/zbs/rbs/zbc`` are sparse
        ``{"m", "n", "value"}`` lists; axis arrays are dense.  Unknown keys
        (e.g. VMEC++ ``free_boundary_method``) are ignored.
        """
        data = json.loads(text)
        if "adiabatic_index" in data and "gamma" not in data:
            data["gamma"] = data["adiabatic_index"]

        mpol = int(data.get("mpol", 6))
        ntor = int(data.get("ntor", 0))

        def boundary(name: str) -> np.ndarray | None:
            entries = data.get(name)
            if entries is None:
                return None
            grid = np.zeros((2 * ntor + 1, mpol))
            for entry in entries:
                n, m = int(entry["n"]), int(entry["m"])
                if -ntor <= n <= ntor and 0 <= m < mpol:
                    grid[n + ntor, m] = float(entry["value"])
            return grid

        kwargs: Dict[str, Any] = {}
        field_names = {f.name for f in fields(cls)}
        for name in field_names - {"rbc", "zbs", "rbs", "zbc"}:
            if name in data and data[name] is not None:
                kwargs[name] = data[name]
        for name in ("rbc", "zbs", "rbs", "zbc"):
            grid = boundary(name)
            if grid is not None:
                kwargs[name] = grid
        kwargs["mpol"] = mpol
        kwargs["ntor"] = ntor
        if "lfreeb" not in kwargs:
            kwargs["lfreeb"] = False  # VMEC++ default (vmecpp.VmecInput)
        return cls(**kwargs)

    # -- writers --------------------------------------------------------------

    def to_json(self, path: str | Path) -> Path:
        """Write VMEC++-schema JSON that round-trips through :meth:`from_file`.

        Boundary coefficients are written as sparse ``{"m","n","value"}``
        lists (nonzero entries only); axis and profile arrays are dense.
        """
        def sparse(grid: np.ndarray) -> list:
            entries = []
            for n_shift, m in zip(*np.nonzero(grid)):
                entries.append({"m": int(m), "n": int(n_shift) - self.ntor,
                                "value": float(grid[n_shift, m])})
            return entries

        data: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("rbc", "zbs", "rbs", "zbc"):
                data[f.name] = sparse(value)
            elif isinstance(value, np.ndarray):
                data[f.name] = value.tolist()
            else:
                data[f.name] = value
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1) + "\n")
        return path

    def to_indata(self, path: str | Path) -> Path:
        """Write a classic ``&INDATA`` namelist that round-trips exactly.

        Floats are written with 17 significant digits so re-parsing
        reproduces the same binary values; empty arrays are omitted.
        """
        def fmt(value) -> str:
            if isinstance(value, (bool, np.bool_)):
                return ".TRUE." if value else ".FALSE."
            if isinstance(value, (int, np.integer)):
                return str(int(value))
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.17E}"
            return "'" + str(value).replace("'", "''") + "'"

        lines: List[str] = ["&INDATA"]

        def put(name: str, value) -> None:
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return
                lines.append(f"  {name} = " + ", ".join(fmt(v) for v in value.tolist()))
            else:
                lines.append(f"  {name} = {fmt(value)}")

        put("LASYM", self.lasym)
        put("NFP", self.nfp)
        put("MPOL", self.mpol)
        put("NTOR", self.ntor)
        put("NTHETA", self.ntheta)
        put("NZETA", self.nzeta)
        put("NS_ARRAY", self.ns_array)
        put("FTOL_ARRAY", self.ftol_array)
        put("NITER_ARRAY", self.niter_array)
        put("DELT", self.delt)
        put("TCON0", self.tcon0)
        put("APHI", self.aphi)
        put("PHIEDGE", self.phiedge)
        put("NSTEP", self.nstep)
        put("GAMMA", self.gamma)
        put("SPRES_PED", self.spres_ped)
        put("PRES_SCALE", self.pres_scale)
        put("PMASS_TYPE", self.pmass_type)
        put("AM", self.am)
        put("AM_AUX_S", self.am_aux_s)
        put("AM_AUX_F", self.am_aux_f)
        put("NCURR", self.ncurr)
        put("CURTOR", self.curtor)
        put("PCURR_TYPE", self.pcurr_type)
        put("AC", self.ac)
        put("AC_AUX_S", self.ac_aux_s)
        put("AC_AUX_F", self.ac_aux_f)
        put("PIOTA_TYPE", self.piota_type)
        put("AI", self.ai)
        put("AI_AUX_S", self.ai_aux_s)
        put("AI_AUX_F", self.ai_aux_f)
        put("BLOAT", self.bloat)
        put("LFREEB", self.lfreeb)
        put("MGRID_FILE", self.mgrid_file)
        put("EXTCUR", self.extcur)
        put("NVACSKIP", self.nvacskip)
        put("MFILTER_FBDY", self.mfilter_fbdy)
        put("NFILTER_FBDY", self.nfilter_fbdy)
        put("PRECON_TYPE", self.precon_type)
        put("PREC2D_THRESHOLD", self.prec2d_threshold)
        put("RAXIS_CC", self.raxis_c)
        put("ZAXIS_CS", self.zaxis_s)
        if self.lasym or np.any(self.raxis_s) or np.any(self.zaxis_c):
            put("RAXIS_CS", self.raxis_s)
            put("ZAXIS_CC", self.zaxis_c)
        for name, grid in (("RBC", self.rbc), ("ZBS", self.zbs),
                           ("RBS", self.rbs), ("ZBC", self.zbc)):
            for n_shift, m in zip(*np.nonzero(grid)):
                lines.append(
                    f"  {name}({int(n_shift) - self.ntor},{int(m)}) = "
                    f"{fmt(float(grid[n_shift, m]))}"
                )
        lines.append("/")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path
