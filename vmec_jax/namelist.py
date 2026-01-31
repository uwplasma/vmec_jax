"""Minimal parser for VMEC-style Fortran namelist (&INDATA).

Goals:
- **No third-party dependency** (no f90nml)
- Works on VMEC2000 input files used in the included python/tests
- Handles indexed coefficient syntax: RBC(m,n)=..., including negative indices
- Handles arrays split across multiple lines

This is not a full Fortran namelist implementation, but is intentionally targeted to VMEC inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

Number = Union[int, float]
Scalar = Union[str, bool, Number]
Value = Union[Scalar, List[Scalar]]


def _strip_fortran_comments(line: str) -> str:
    """Remove '!' comments, respecting single-quoted strings."""
    out = []
    in_quote = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'":
            in_quote = not in_quote
            out.append(ch)
        elif ch == "!" and not in_quote:
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_ASSIGN_RE = re.compile(r"(?P<key>[A-Za-z_]\w*(?:\([^\)]*\))?)\s*=", re.MULTILINE)


def _tokenize_value_chunk(chunk: str) -> List[str]:
    """Tokenize a value chunk into tokens, keeping quoted strings intact."""
    tokens: List[str] = []
    buf: List[str] = []
    in_quote = False
    chunk = chunk.strip()
    i = 0
    while i < len(chunk):
        ch = chunk[i]
        if ch == "'":
            in_quote = not in_quote
            buf.append(ch)
        elif not in_quote and ch in [",", "\n", "\t", " ", "\r"]:
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    tokens.append(tok)
                buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        tok = "".join(buf).strip()
        if tok:
            tokens.append(tok)
    return tokens


_BOOL_TRUE = {"T", ".T.", ".TRUE.", "TRUE"}
_BOOL_FALSE = {"F", ".F.", ".FALSE.", "FALSE"}


def _parse_scalar(tok: str) -> Scalar:
    tok = tok.strip()
    # strings
    if len(tok) >= 2 and tok[0] == "'" and tok[-1] == "'":
        return tok[1:-1]
    up = tok.upper()
    if up in _BOOL_TRUE:
        return True
    if up in _BOOL_FALSE:
        return False
    # integers (including leading zeros)
    if re.fullmatch(r"[+-]?\d+", tok):
        try:
            return int(tok)
        except Exception:
            pass
    # floats (Fortran exponent forms)
    # Accept D or E exponent.
    f = tok.replace("D", "E").replace("d", "E")
    try:
        return float(f)
    except Exception:
        # fall back to raw string
        return tok


def _parse_key(key: str) -> Tuple[str, Tuple[int, ...] | None]:
    """Split KEY or KEY(i,j) into (base, indices)."""
    key = key.strip()
    if "(" not in key:
        return key.upper(), None
    base, rest = key.split("(", 1)
    rest = rest.rstrip(")")
    idx = tuple(int(x.strip()) for x in rest.split(",") if x.strip() != "")
    return base.upper(), idx


@dataclass
class InData:
    scalars: Dict[str, Value]
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]]

    def get(self, name: str, default: Value | None = None) -> Value | None:
        return self.scalars.get(name.upper(), default)

    def get_bool(self, name: str, default: bool = False) -> bool:
        v = self.get(name, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, list) and v and isinstance(v[0], bool):
            return bool(v[0])
        return bool(v)

    def get_int(self, name: str, default: int = 0) -> int:
        v = self.get(name, default)
        if isinstance(v, list):
            v = v[0] if v else default
        try:
            return int(v)  # type: ignore[arg-type]
        except Exception:
            return default

    def get_float(self, name: str, default: float = 0.0) -> float:
        v = self.get(name, default)
        if isinstance(v, list):
            v = v[0] if v else default
        try:
            return float(v)  # type: ignore[arg-type]
        except Exception:
            return default


def read_indata(path: str | Path) -> InData:
    """Read &INDATA from a VMEC input file."""
    text = Path(path).read_text()
    # isolate &INDATA block
    m_start = re.search(r"&\s*INDATA", text, flags=re.IGNORECASE)
    if not m_start:
        raise ValueError("No &INDATA found")
    # naive end: first '/' after start
    m_end = re.search(r"\n\s*/\s*\n|\n\s*/\s*$", text[m_start.end() :], flags=re.MULTILINE)
    if not m_end:
        raise ValueError("No terminating '/' for &INDATA")
    block = text[m_start.end() : m_start.end() + m_end.start()]

    # strip comments line-by-line
    lines = [_strip_fortran_comments(ln) for ln in block.splitlines()]
    cleaned = "\n".join(lines)

    scalars: Dict[str, Value] = {}
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]] = {}

    matches = list(_ASSIGN_RE.finditer(cleaned))
    if not matches:
        return InData(scalars=scalars, indexed=indexed)

    for i, m in enumerate(matches):
        key_raw = m.group("key")
        key_base, idx = _parse_key(key_raw)
        val_start = m.end()
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        chunk = cleaned[val_start:val_end].strip()
        # remove trailing commas
        chunk = re.sub(r",\s*$", "", chunk)
        toks = _tokenize_value_chunk(chunk)
        parsed = [_parse_scalar(t) for t in toks]
        value: Value
        if len(parsed) == 0:
            continue
        elif len(parsed) == 1:
            value = parsed[0]
        else:
            value = parsed

        if idx is None:
            scalars[key_base] = value
        else:
            if key_base not in indexed:
                indexed[key_base] = {}
            # indexed assignments in VMEC are scalar
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(f"Indexed assignment {key_raw} has multiple values")
                indexed[key_base][idx] = value[0]
            else:
                indexed[key_base][idx] = value

    return InData(scalars=scalars, indexed=indexed)
