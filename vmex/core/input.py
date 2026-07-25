"""VMEC input handling: the ``&INDATA`` namelist and VMEC++-style JSON.

VMEC2000 counterparts: ``LIBSTELL/Sources/Modules/vmec_input.f``
(``read_indata_namelist``: variable set and defaults) and ``readin.f``
(post-read normalizations).  The JSON schema follows VMEC++
(``vmecpp.VmecInput``): identical key names, boundary coefficients as sparse
``{"m": int, "n": int, "value": float}`` lists, dense axis arrays, and
``adiabatic_index`` accepted as an alias for ``gamma``.

:class:`VmecInput` is a frozen dataclass holding the INDATA content this code
base actually consumes, with VMEC2000 defaults.  Parsing is host-side NumPy
code (nothing here needs JAX).  Controls which would change the mathematical
problem or iteration contract but are not implemented are rejected by
:class:`UnsupportedInputModeError`; they are never silently converted into an
ordinary fixed-/free-boundary solve.

Normalizations applied on construction (all from VMEC2000):

* ``read_indata_namelist``: ``raxis_s[0] = 0`` and ``zaxis_s[0] = 0``; the
  obsolete ``RAXIS``/``ZAXIS`` arrays override ``RAXIS_CC``/``ZAXIS_CS`` where
  nonzero; ``niter_array`` falls back to ``NITER`` when absent.
* ``readin.f``: the explicit legacy ``NS_ARRAY(1)=0`` form expands to
  ``[max(3, NSIN), 31]``; ``lfreeb`` is forced ``False`` when
  ``mgrid_file == 'NONE'``; ``nvacskip <= 0`` falls back to ``nfp``.
* Boundary coefficients outside ``|n| <= ntor``, ``0 <= m < mpol`` are
  dropped (VMEC2000 reads them into oversized arrays but never uses them).

Index conventions: ``rbc/zbs/rbs/zbc`` are dense 2D arrays of shape
``(2*ntor + 1, mpol)`` indexed ``[n + ntor, m]``, i.e. ``rbc[n + ntor, m]``
is the INDATA coefficient ``RBC(n, m)``.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

__all__ = ["UnsupportedInputModeError", "VmecInput"]

Scalar = Union[str, bool, int, float]
IndexComponent = Union[int, slice]

# Declared VMEC2000 namelist bounds for arrays consumed by VMEX or inspected
# by the active-mode classifier.  Bounds are inclusive and ordered exactly as
# in Fortran; the first subscript therefore varies fastest during namelist
# sequence association.  vmec_input.f uses ntord=101, mpol1d=100,
# ndatafmax=101, and nigroup=300.
_INDATA_ARRAY_BOUNDS: dict[str, tuple[tuple[int, int], ...]] = {
    "NS_ARRAY": ((1, 100),),
    "NITER_ARRAY": ((1, 100),),
    "FTOL_ARRAY": ((1, 100),),
    "APHI": ((1, 20),),
    **{name: ((0, 20),) for name in ("AM", "AI", "AC", "AH", "AT")},
    **{
        name: ((1, 101),)
        for name in (
            "AM_AUX_S", "AM_AUX_F", "AI_AUX_S", "AI_AUX_F",
            "AC_AUX_S", "AC_AUX_F", "AH_AUX_S", "AH_AUX_F",
            "AT_AUX_S", "AT_AUX_F", "PSA", "PFA", "ISA", "IFA",
        )
    },
    **{
        name: ((0, 101),)
        for name in (
            "RAXIS", "ZAXIS", "RAXIS_CC", "RAXIS_CS",
            "ZAXIS_CC", "ZAXIS_CS",
        )
    },
    **{
        name: ((-101, 101), (0, 100))
        for name in ("RBC", "ZBS", "RBS", "ZBC")
    },
    "EXTCUR": ((1, 300),),
    "BOOZ_SURFACES": ((1, 10001),),
}


class UnsupportedInputModeError(ValueError):
    """An input requests semantics which VMEX does not implement.

    ``code`` is a stable, value-free diagnostic code suitable for the
    privacy-preserving input checker.  ``control`` names the INDATA/JSON
    control without echoing its value or any equilibrium data.
    """

    def __init__(self, code: str, control: str, reason: str):
        self.code = str(code)
        self.control = str(control)
        self.reason = str(reason)
        super().__init__(f"{self.control}: {self.reason}")

# ---------------------------------------------------------------------------
# Tolerant Fortran-namelist tokenizer (targeted at VMEC &INDATA files)
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(r"(?P<key>[A-Za-z_]\w*(?:\([^\)]*\))?)\s*=", re.MULTILINE)
_REPEAT_RE = re.compile(r"^(?P<count>\d+)\*(?P<value>.+)$")
_BOOL_TRUE = {"T", ".T.", ".TRUE.", "TRUE"}
_BOOL_FALSE = {"F", ".F.", ".FALSE.", "FALSE"}

# Complete &INDATA name inventory from
# STELLOPT/LIBSTELL/Sources/Modules/vmec_input.f, plus the documented VMEX
# Boozer post-processing extension.  Values may be supported, explicitly
# rejected when active, or accepted as output-only compatibility controls;
# an unclassified spelling is always an input error.
_KNOWN_INDATA_NAMES = {
    "MGRID_FILE", "TIME_SLICE", "NFP", "NCURR", "NSIN", "NITER", "NSTEP",
    "NVACSKIP", "DELT", "FTOL", "GAMMA", "BLOAT", "AM", "AI", "AC", "APHI",
    "PCURR_TYPE", "PMASS_TYPE", "PIOTA_TYPE",
    "AM_AUX_S", "AM_AUX_F", "AI_AUX_S", "AI_AUX_F", "AC_AUX_S", "AC_AUX_F",
    "AH", "AT", "BCRIT", "PH_TYPE", "AH_AUX_S", "AH_AUX_F",
    "PT_TYPE", "AT_AUX_S", "AT_AUX_F",
    "RBC", "ZBS", "RBS", "ZBC", "SPRES_PED", "PRES_SCALE",
    "RAXIS_CC", "ZAXIS_CS", "RAXIS_CS", "ZAXIS_CC", "RAXIS", "ZAXIS",
    "MPOL", "NTOR", "NTHETA", "NZETA", "MFILTER_FBDY", "NFILTER_FBDY",
    "NITER_ARRAY", "PRE_NITER", "NS_ARRAY", "FTOL_ARRAY", "TCON0",
    "PRECON_TYPE", "PREC2D_THRESHOLD", "CURTOR", "SIGMA_CURRENT", "EXTCUR",
    "OMP_NUM_THREADS", "PHIEDGE",
    "PSA", "PFA", "ISA", "IFA", "IMATCH_PHIEDGE", "IOPT_RAXIS",
    "TENSI", "TENSP", "MSEANGLE_OFFSET", "MSEANGLE_OFFSETM", "IMSE",
    "ISNODES", "RSTARK", "DATASTARK", "SIGMA_STARK", "ITSE", "IPNODES",
    "PRESFAC", "PRES_OFFSET", "RTHOM", "DATATHOM", "SIGMA_THOM", "PHIDIAM",
    "SIGMA_DELPHID", "TENSI2", "FPOLYI", "NFLXS", "INDXFLX", "DSIOBT",
    "SIGMA_FLUX", "NBFLD", "INDXBFLD", "BBC", "SIGMA_B", "LPOFR",
    "LFORBAL", "LFREEB", "LMOVE_AXIS", "LRECON", "LMAC", "LMOVIE",
    "LASYM", "LEDGE_DUMP", "LSPECTRUM_DUMP", "LOPTIM", "LRFP",
    "LOLDOUT", "LWOUTTXT", "LDIAGNO", "LFULL3D1OUT",
    "MAX_MAIN_ITERATIONS", "LGIVEUP", "FGIVEUP", "LBSUBS", "TRIP3D_FILE",
    "LNYQUIST", "TVOLUME", "LVOLUME_RFIX",
    # VMEX post-processing spellings found in the repository's input decks.
    "LBOOZ", "MBOOZ", "NBOOZ", "BOOZ_SURFACES",
}


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


def _parse_key(key: str) -> Tuple[str, Tuple[IndexComponent, ...] | None]:
    """Split a namelist key into its name and scalar/section designator.

    A bare key returns ``None``.  Fortran triplets, including ``KEY(:)``,
    retain their inclusive upper bound in a :class:`slice`; they are expanded
    against the declared VMEC2000 bounds after tokenization.
    """
    key = key.strip()
    if "(" not in key:
        return key.upper(), None
    base, rest = key.split("(", 1)
    rest = rest.rstrip(")")
    components: list[IndexComponent] = []
    for component in rest.split(","):
        component = component.strip()
        if ":" not in component:
            components.append(int(component))
            continue
        parts = component.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid namelist array section: {key}")
        start = int(parts[0]) if parts[0].strip() else None
        stop = int(parts[1]) if parts[1].strip() else None
        step = int(parts[2]) if len(parts) == 3 and parts[2].strip() else 1
        if step == 0:
            raise ValueError(f"zero stride in namelist array section: {key}")
        components.append(slice(start, stop, step))
    return base.upper(), tuple(components)


def _fortran_positions(
    name: str,
    designator: Tuple[IndexComponent, ...] | None,
    value_count: int,
) -> list[tuple[int, ...]]:
    """Expand one array assignment in Fortran namelist element order."""
    bounds = _INDATA_ARRAY_BOUNDS.get(name)
    if bounds is None:
        if designator is None:
            raise ValueError(f"array bounds unavailable for INDATA variable {name}")
        if any(isinstance(component, slice) for component in designator):
            raise ValueError(f"array bounds unavailable for INDATA section {name}")
        # Explicit scalar subscripts remain usable for compatibility-only
        # arrays whose compile-time extents do not affect the VMEX solve.
        if value_count != 1:
            raise ValueError(
                f"multivalue starting-element assignment needs declared bounds: {name}"
            )
        return [tuple(int(component) for component in designator)]

    rank = len(bounds)
    if designator is None:
        designator = tuple(slice(None, None, 1) for _ in bounds)
    if len(designator) != rank:
        raise ValueError(
            f"wrong number of subscripts for {name}: expected {rank}, "
            f"got {len(designator)}"
        )

    if all(not isinstance(component, slice) for component in designator):
        start = tuple(int(component) for component in designator)
        sizes = [upper - lower + 1 for lower, upper in bounds]
        offset = 0
        stride = 1
        for component, (lower, upper), size in zip(
            start, bounds, sizes, strict=True
        ):
            if not lower <= component <= upper:
                raise ValueError(f"array subscript outside declared bounds: {name}")
            offset += (component - lower) * stride
            stride *= size
        if offset + value_count > stride:
            raise ValueError(f"too many values for namelist array assignment: {name}")
        positions: list[tuple[int, ...]] = []
        for linear in range(offset, offset + value_count):
            remainder = linear
            position: list[int] = []
            for (lower, _upper), size in zip(bounds, sizes, strict=True):
                position.append(lower + remainder % size)
                remainder //= size
            positions.append(tuple(position))
        return positions

    axes: list[list[int]] = []
    for component, (lower, upper) in zip(designator, bounds, strict=True):
        if not isinstance(component, slice):
            value = int(component)
            if not lower <= value <= upper:
                raise ValueError(f"array subscript outside declared bounds: {name}")
            axes.append([value])
            continue
        step = 1 if component.step is None else int(component.step)
        start = (
            lower if step > 0 else upper
        ) if component.start is None else int(component.start)
        stop = (
            upper if step > 0 else lower
        ) if component.stop is None else int(component.stop)
        if not lower <= start <= upper or not lower <= stop <= upper:
            raise ValueError(f"array section outside declared bounds: {name}")
        exclusive = stop + (1 if step > 0 else -1)
        axes.append(list(range(start, exclusive, step)))

    positions = [
        tuple(reversed(position))
        for position in product(*reversed(axes))
    ]
    if value_count > len(positions):
        raise ValueError(f"too many values for namelist array section: {name}")
    return positions[:value_count]


def _read_indata_text(text: str) -> tuple[Dict[str, List[Scalar]], Dict[str, Dict[Tuple[int, ...], Scalar]]]:
    """Parse and sequentially replay the ``&INDATA`` block.

    Returns ``(scalars, indexed)`` where ``scalars`` maps upper-case names to
    their final scalar token and ``indexed`` contains the final value of every
    explicitly assigned array element.  Replaying assignments in source order
    reproduces Fortran overlay semantics for repeated dense/indexed writes.
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
        is_array = name in _INDATA_ARRAY_BOUNDS or idx is not None
        if not is_array:
            if len(values) != 1:
                raise ValueError(f"too many values for scalar INDATA variable: {name}")
            scalars[name] = values
        else:
            entries = indexed.setdefault(name, {})
            positions = _fortran_positions(name, idx, len(values))
            for position, value in zip(positions, values, strict=True):
                entries[position] = value
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


def _vmec_ns_prefix(values: Any) -> np.ndarray:
    """Return the positive nondecreasing prefix selected by ``readin.f``."""
    ns = np.atleast_1d(np.asarray(values, dtype=np.int64)).ravel()
    end = 0
    previous = 1
    for value in ns:
        if int(value) <= 0 or int(value) < previous:
            break
        previous = max(previous, int(value))
        end += 1
    return ns[:end].copy()


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


def _indata_values(
    name: str,
    scalars: Dict[str, List[Scalar]],
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]],
) -> list[Scalar]:
    """All explicitly assigned values for one INDATA variable."""
    return list(scalars.get(name, ())) + list(indexed.get(name, {}).values())


def _validate_indata_modes(
    scalars: Dict[str, List[Scalar]],
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]],
) -> None:
    """Reject active VMEC2000 modes that VMEX cannot faithfully execute.

    This check deliberately runs before :class:`VmecInput` construction:
    parsing a switch is not evidence that the production force/iteration path
    honors it.  Neutral legacy spellings remain accepted so standard VMEC2000
    decks need not be edited merely to remove default-valued controls.
    """

    unknown = sorted((set(scalars) | set(indexed)) - _KNOWN_INDATA_NAMES)
    if unknown:
        raise ValueError("unknown INDATA variable(s): " + ", ".join(unknown))

    def first(name: str, default: Scalar) -> Scalar:
        values = scalars.get(name)
        return values[0] if values else default

    reconstruction_active = bool(first("LRECON", True)) and (
        int(first("ITSE", 0)) > 0 or int(first("IMSE", -1)) > 0
    )
    if reconstruction_active:
        raise UnsupportedInputModeError(
            "D00A_RECONSTRUCTION_MODE_UNSUPPORTED",
            "LRECON/ITSE/IMSE",
            "equilibrium reconstruction is not implemented",
        )
    if bool(first("LRFP", False)):
        raise UnsupportedInputModeError(
            "D00B_RFP_MODE_UNSUPPORTED",
            "LRFP",
            "reversed-field-pinch profile and vacuum semantics are not implemented",
        )

    trip3d_file = str(first("TRIP3D_FILE", "NONE")).strip().strip("'\"")
    if trip3d_file.upper() not in {"", "NONE"}:
        raise UnsupportedInputModeError(
            "D00E_TRIP3D_MODE_UNSUPPORTED",
            "TRIP3D_FILE",
            "TRIP3D external-field coupling is not implemented",
        )

    ah_active = any(float(value) != 0.0 for value in _indata_values("AH", scalars, indexed))
    at_active = False
    dense_at = scalars.get("AT", ())
    if dense_at:
        at_active = float(dense_at[0]) != 1.0 or any(
            float(value) != 0.0 for value in dense_at[1:]
        )
    for position, value in indexed.get("AT", {}).items():
        if len(position) == 1:
            expected = 1.0 if position[0] == 0 else 0.0
            at_active = at_active or float(value) != expected
    if ah_active or at_active:
        raise UnsupportedInputModeError(
            "D00F_ANIMEC_MODE_UNSUPPORTED",
            "AH/AT",
            "anisotropic-pressure/flow (ANIMEC) physics is not implemented",
        )

    if float(first("TVOLUME", -1.0)) > 0.0:
        raise UnsupportedInputModeError(
            "D00G_VOLUME_RESCALE_UNSUPPORTED",
            "TVOLUME/LVOLUME_RFIX",
            "VMEC2000 boundary-volume rescaling is not implemented",
        )

    precon_type = str(first("PRECON_TYPE", "NONE")).strip().upper()
    if precon_type not in {"", "NONE", "DEFAULT", "GMRES"}:
        raise UnsupportedInputModeError(
            "D00H_PRECONDITIONER_MODE_UNSUPPORTED",
            "PRECON_TYPE",
            "only NONE/DEFAULT (1-D) and VMEX matrix-free GMRES are implemented",
        )
    if precon_type == "GMRES" and int(first("PRE_NITER", -1)) != -1:
        raise UnsupportedInputModeError(
            "D00I_ITERATION_CONTROL_UNSUPPORTED",
            "PRE_NITER",
            "the VMEC2000 post-activation iteration-budget override is not implemented",
        )
    if int(first("MAX_MAIN_ITERATIONS", 1)) > 1:
        raise UnsupportedInputModeError(
            "D00I_ITERATION_CONTROL_UNSUPPORTED",
            "MAX_MAIN_ITERATIONS",
            "automatic continuation by additional NITER blocks is not implemented",
        )
    if bool(first("LGIVEUP", False)):
        raise UnsupportedInputModeError(
            "D00I_ITERATION_CONTROL_UNSUPPORTED",
            "LGIVEUP/FGIVEUP",
            "VMEC2000 early termination between multigrid stages is not implemented",
        )

    if bool(first("LBSUBS", False)):
        raise UnsupportedInputModeError(
            "D00J_OUTPUT_MODE_UNSUPPORTED",
            "LBSUBS",
            "the alternative VMEC2000 B_s WOUT diagnostic is not implemented",
        )
    if not bool(first("LNYQUIST", True)):
        raise UnsupportedInputModeError(
            "D00J_OUTPUT_MODE_UNSUPPORTED",
            "LNYQUIST",
            "suppressing Nyquist WOUT tables is not implemented",
        )
    if bool(first("LBOOZ", False)):
        raise UnsupportedInputModeError(
            "D00J_OUTPUT_MODE_UNSUPPORTED",
            "LBOOZ",
            "INDATA-driven Boozer output is not implemented; use the --booz CLI option",
        )
    requested_legacy_artifacts = [
        name
        for name in ("LMAC", "LEDGE_DUMP", "LOLDOUT", "LWOUTTXT", "LDIAGNO")
        if bool(first(name, False))
    ]
    if requested_legacy_artifacts:
        controls = "/".join(requested_legacy_artifacts)
        warnings.warn(
            f"{controls}: requested VMEC2000 auxiliary output artifact is not "
            "implemented; the equilibrium solve will continue",
            RuntimeWarning,
            stacklevel=3,
        )


def _validate_json_modes(data: Dict[str, Any]) -> None:
    """Apply the same no-silent-fallback policy to VMEC++ JSON controls."""
    method = str(data.get("free_boundary_method", "nestor")).strip().lower()
    if method != "nestor":
        raise UnsupportedInputModeError(
            "D00K_FREE_BOUNDARY_METHOD_UNSUPPORTED",
            "free_boundary_method",
            "VMEX JSON solves support NESTOR; only_coils/biest are different models",
        )

    field_names = {field.name for field in fields(VmecInput)}
    known = field_names | {"adiabatic_index", "free_boundary_method"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            "unknown VMEC++ JSON input key(s): " + ", ".join(unknown)
        )


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
    lforbal: bool = False        #: replace m=1,n=0 R/Z forces by average force balance
    lmove_axis: bool = True      #: improve the axis when the first force sum is > 1e2
    lfull3d1out: bool = False    #: write a WOUT when the iteration limit is reached
    aphi: Any = None             #: radial-flux remap polynomial (default [1,0,...], len 20)
    phiedge: float = 1.0         #: total enclosed toroidal flux [Wb]
    nstep: int = 10              #: iterations between progress prints
    time_slice: float = 0.0      #: informational value in the VMEC run header

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
        set_(self, "lforbal", bool(self.lforbal))
        set_(self, "lmove_axis", bool(self.lmove_axis))
        set_(self, "lfull3d1out", bool(self.lfull3d1out))
        for name in ("nfp", "mpol", "ntor", "ntheta", "nzeta", "ncurr", "nstep",
                     "nvacskip", "mfilter_fbdy", "nfilter_fbdy"):
            set_(self, name, int(getattr(self, name)))
        for name in ("delt", "tcon0", "phiedge", "time_slice", "pres_scale",
                     "gamma", "spres_ped", "curtor", "bloat",
                     "prec2d_threshold"):
            set_(self, name, float(getattr(self, name)))
        for name in ("pmass_type", "pcurr_type", "piota_type"):
            set_(self, name, str(getattr(self, name)).strip().lower())
        set_(self, "precon_type", str(self.precon_type).strip())
        set_(self, "mgrid_file", str(self.mgrid_file).strip())

        # readin.f stops at the first nonpositive or decreasing entry; later
        # values are outside multi_ns_grid and never reach runvmec.f.
        ns = _vmec_ns_prefix([31] if self.ns_array is None else self.ns_array)
        if ns.size == 0:
            ns = np.asarray([31], dtype=np.int64)
        n_stages = int(ns.size)
        set_(self, "ns_array", ns)
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
        _validate_indata_modes(scalars, indexed)

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

        def vector(
            name: str,
            *,
            lower: int,
            default: list[float] | np.ndarray | None = None,
            size: int | None = None,
        ) -> np.ndarray:
            """Apply dense and indexed namelist assignments to one vector.

            Fortran namelist reads overlay assignments on the values initialized
            by ``read_indata_namelist``.  In particular, ``APHI(2)=...`` must
            retain VMEC's default ``APHI(1)=1``.  The old parser consumed only
            unindexed vector assignments, silently ignoring indexed profile and
            multigrid entries.
            """
            dense = get_list(name) or []
            entries = indexed.get(name, {})
            indexed_length = max(
                (idx[0] - lower + 1 for idx in entries if len(idx) == 1),
                default=0,
            )
            default_values = _float_array(default)
            length = size if size is not None else max(
                len(dense), indexed_length, int(default_values.size)
            )
            out = _fixed_length(default_values, length) if length else np.zeros((0,))
            if dense:
                count = min(len(dense), length)
                out[:count] = np.asarray(dense[:count], dtype=float)
            for idx, value in entries.items():
                if len(idx) != 1:
                    continue
                position = idx[0] - lower
                if 0 <= position < length:
                    out[position] = float(value)
            return out

        mpol = int(get("MPOL", 6))
        ntor = int(get("NTOR", 0))

        # vmec_input.f initializes NS_ARRAY(1)=31.  readin.f's explicitly
        # requested old-style sentinel NS_ARRAY(1)=0 expands to NSIN then 31.
        ns_array = vector("NS_ARRAY", lower=1, default=[31])
        if ns_array.size and int(ns_array[0]) == 0:
            ns_array = np.asarray([max(3, int(get("NSIN", 31))), 31], dtype=np.int64)
        ns_array = _vmec_ns_prefix(ns_array)
        if ns_array.size == 0:
            ns_array = np.asarray([31], dtype=np.int64)
        n_stages = int(ns_array.size)

        # ftol_array: FTOL_ARRAY, or scalar FTOL (vmec_input.f: ftol_array(1)=ftol).
        ftol_default = np.zeros((n_stages,))
        ftol_default[0] = float(get("FTOL", 1e-10))
        ftol_array = vector(
            "FTOL_ARRAY", lower=1, default=ftol_default
        )
        if ftol_array.size and float(ftol_array[0]) == 0.0:
            target_ftol = float(get("FTOL", 1e-10))
            if n_stages == 1:
                ftol_array[0] = target_ftol
            else:
                stage = np.arange(n_stages, dtype=float)
                ftol_array[:n_stages] = 1.0e-8 * (
                    1.0e8 * target_ftol
                ) ** (stage / float(n_stages - 1))
        # vmec_input.f initializes every NITER_ARRAY entry to -1, and replaces
        # the complete array with NITER only when no element was assigned.
        niter_assigned = "NITER_ARRAY" in scalars or "NITER_ARRAY" in indexed
        niter_default = np.full((n_stages,), -1 if niter_assigned else int(get("NITER", 100)))
        niter_array = vector(
            "NITER_ARRAY", lower=1, default=niter_default
        )

        def axis(name: str, legacy: str | None = None) -> np.ndarray:
            out = _fixed_length(get_list(name) or [], ntor + 1)
            for idx, value in indexed.get(name, {}).items():
                if len(idx) == 1 and 0 <= idx[0] <= ntor:
                    out[idx[0]] = float(value)
            if legacy is not None:
                # Backwards compatibility (read_indata_namelist):
                # WHERE (raxis /= 0) raxis_cc = raxis (idem zaxis -> zaxis_cs).
                old = _fixed_length(get_list(legacy) or [], ntor + 1)
                for idx, value in indexed.get(legacy, {}).items():
                    if len(idx) == 1 and 0 <= idx[0] <= ntor:
                        old[idx[0]] = float(value)
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

        extcur = vector("EXTCUR", lower=1)

        aphi_default = np.zeros((20,))
        aphi_default[0] = 1.0

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
            lforbal=bool(get("LFORBAL", False)),
            lmove_axis=bool(get("LMOVE_AXIS", True)),
            lfull3d1out=bool(get("LFULL3D1OUT", False)),
            aphi=vector("APHI", lower=1, default=aphi_default, size=20),
            phiedge=float(get("PHIEDGE", 1.0)),
            nstep=int(get("NSTEP", 10)),
            time_slice=float(get("TIME_SLICE", 0.0)),
            pmass_type=str(get("PMASS_TYPE", "power_series")),
            am=vector("AM", lower=0, default=np.zeros((21,)), size=21),
            am_aux_s=vector("AM_AUX_S", lower=1),
            am_aux_f=vector("AM_AUX_F", lower=1),
            pres_scale=float(get("PRES_SCALE", 1.0)),
            gamma=float(get("GAMMA", 0.0)),
            spres_ped=float(get("SPRES_PED", 1.0)),
            ncurr=int(get("NCURR", 0)),
            pcurr_type=str(get("PCURR_TYPE", "power_series")),
            ac=vector("AC", lower=0, default=np.zeros((21,)), size=21),
            ac_aux_s=vector("AC_AUX_S", lower=1),
            ac_aux_f=vector("AC_AUX_F", lower=1),
            curtor=float(get("CURTOR", 0.0)),
            piota_type=str(get("PIOTA_TYPE", "power_series")),
            ai=vector("AI", lower=0, default=np.zeros((21,)), size=21),
            ai_aux_s=vector("AI_AUX_S", lower=1),
            ai_aux_f=vector("AI_AUX_F", lower=1),
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
        ``{"m", "n", "value"}`` lists; axis arrays are dense.  The VMEC++
        ``free_boundary_method="nestor"`` spelling is accepted.  Other
        free-boundary methods and unknown keys fail explicitly instead of
        being silently ignored.
        """
        data = json.loads(text)
        _validate_json_modes(data)
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
        put("LFORBAL", self.lforbal)
        put("LMOVE_AXIS", self.lmove_axis)
        put("LFULL3D1OUT", self.lfull3d1out)
        put("APHI", self.aphi)
        put("PHIEDGE", self.phiedge)
        put("NSTEP", self.nstep)
        put("TIME_SLICE", self.time_slice)
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
