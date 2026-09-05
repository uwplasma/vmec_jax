"""VMEX execution metadata carried outside the VMEC physics schema.

VMEX-only run controls travel in places every legacy reader ignores: comment
directives in ``&INDATA`` text and a reserved ``_vmex`` section in structured
JSON.  VMEC2000 treats the directive lines as comments and solves the ordinary
input; a VMEC++-compatible JSON reader sees the physics schema once ``_vmex``
is removed.  :class:`~vmex.core.input.VmecInput` stays a pure physics object —
these options control *execution*, not equilibrium physics, so they never
become dataclass fields on it.

Two directive spellings are accepted, both VMEC-safe comments::

    !@VMEX POLISH = AUTO
    !@VMEX POLISH_TOL = 1.0E-8
    !@VMEX POLISH_FAIL = ERROR
    !@VMEX POLISH_DEGREE = 5
    !@VMEX POLISH_MAX_ITER = 40
    !@VMEX POLISH_SPANS = 16
    !@VMEX POLISH_BUDGET = 3600

and the original single-flag form from the polishing integration::

    ! VMEX: POLISH_FORCE_BALANCE = .TRUE.

Precedence, resolved by :func:`resolve_run_options`, is exactly::

    CLI option > explicit Python keyword > file directive > package default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .errors import VmecInputError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, typing only
    from .input import VmecInput

__all__ = [
    "InputRequest",
    "RunOptions",
    "format_indata_directives",
    "parse_indata_run_options",
    "read_input_request",
    "resolve_run_options",
    "strip_vmex_json",
]

_POLISH_MODES = (False, True, "auto")
_FAIL_MODES = ("error", "fallback", "warn")
_DEGREES = (3, 5, 7)

#: ``!@VMEX KEY = VALUE`` — the canonical directive family.
# [ \t] rather than \s throughout: a greedy \s* would consume the newline
# and let the optional trailing-comment group swallow the next directive line.
_DIRECTIVE = re.compile(
    r"^[ \t]*![ \t]*@[ \t]*VMEX[ \t]+([A-Za-z_]+)[ \t]*=[ \t]*"
    r"([^\s!]+)[ \t]*(?:!.*)?$",
    flags=re.MULTILINE | re.IGNORECASE,
)

#: ``! VMEX: POLISH_FORCE_BALANCE = .TRUE.`` — the original single flag.
_LEGACY_POLISH = re.compile(
    r"^\s*!\s*VMEX\s*:\s*POLISH_FORCE_BALANCE\s*=\s*([^\s,!]+)",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class RunOptions:
    """How to execute a solve; never what equilibrium to solve.

    ``polish`` is ``False``, ``True``, or ``"auto"`` (polish only when the
    legacy solve converged and the physics is in the supported set).
    ``polish_tol``/``polish_degree``/``polish_max_iter``/``polish_spans``
    override the matching :class:`~vmex.core.polish_driver.PolishConfig`
    fields (``tolerance``, ``radial_degree``, ``max_nonlinear_iterations``,
    ``radial_spans``) when set; only knobs that exist on the driver config
    are exposed here.
    ``polish_fail`` maps onto the driver's fail policy: ``"error"`` raises,
    ``"fallback"`` returns the unpolished state silently, ``"warn"`` returns
    it with a :class:`RuntimeWarning`.
    ``polish_budget`` is the wall-clock ceiling in seconds that ``POLISH =
    AUTO`` will commit to (``auto_budget_seconds``); it raises or lowers the
    cost at which AUTO declines to polish and has no effect on ``POLISH =
    ON``, which never consults it.
    """

    polish: bool | str = False
    polish_tol: float | None = None
    polish_fail: str = "error"
    polish_degree: int | None = None
    polish_max_iter: int | None = None
    polish_spans: int | None = None
    polish_budget: float | None = None

    def __post_init__(self) -> None:
        if self.polish not in _POLISH_MODES:
            raise VmecInputError(
                f"POLISH must be one of {_POLISH_MODES}, got {self.polish!r}")
        if self.polish_fail not in _FAIL_MODES:
            raise VmecInputError(
                f"POLISH_FAIL must be one of {_FAIL_MODES}, "
                f"got {self.polish_fail!r}")
        if self.polish_tol is not None and not self.polish_tol > 0.0:
            raise VmecInputError(
                f"POLISH_TOL must be positive, got {self.polish_tol!r}")
        if self.polish_degree is not None and self.polish_degree not in _DEGREES:
            raise VmecInputError(
                f"POLISH_DEGREE must be one of {_DEGREES}, "
                f"got {self.polish_degree!r}")
        if self.polish_max_iter is not None and self.polish_max_iter < 1:
            raise VmecInputError(
                f"POLISH_MAX_ITER must be positive, got {self.polish_max_iter!r}")
        if self.polish_spans is not None and self.polish_spans < 1:
            raise VmecInputError(
                f"POLISH_SPANS must be positive, got {self.polish_spans!r}")
        if self.polish_budget is not None and not self.polish_budget > 0.0:
            raise VmecInputError(
                f"POLISH_BUDGET must be positive, got {self.polish_budget!r}")


@dataclass(frozen=True)
class InputRequest:
    """One parsed input file: the physics and how to run it."""

    input: "VmecInput"
    options: RunOptions
    source: Path


def _parse_polish(token: str, *, key: str) -> bool | str:
    lowered = token.strip().lower().strip(".")
    if lowered in ("auto",):
        return "auto"
    if lowered in ("true", "t", "1", "yes", "on"):
        return True
    if lowered in ("false", "f", "0", "no", "off"):
        return False
    raise VmecInputError(
        f"{key} must be AUTO, TRUE, or FALSE, got {token!r}")


def _parse_directive_value(key: str, token: str) -> tuple[str, Any]:
    """Map one directive assignment onto a :class:`RunOptions` field."""
    if key == "POLISH":
        return "polish", _parse_polish(token, key=key)
    if key == "POLISH_TOL":
        try:
            return "polish_tol", float(token.rstrip(","))
        except ValueError as error:
            raise VmecInputError(
                f"POLISH_TOL must be a real number, got {token!r}") from error
    if key == "POLISH_BUDGET":
        try:
            return "polish_budget", float(token.rstrip(","))
        except ValueError as error:
            raise VmecInputError(
                f"POLISH_BUDGET must be a real number, got {token!r}"
            ) from error
    if key == "POLISH_FAIL":
        return "polish_fail", token.strip().lower()
    if key in ("POLISH_DEGREE", "POLISH_MAX_ITER", "POLISH_SPANS"):
        try:
            return key.lower(), int(token.rstrip(","))
        except ValueError as error:
            raise VmecInputError(
                f"{key} must be an integer, got {token!r}") from error
    raise VmecInputError(
        f"unknown VMEX directive {key!r} "
        "(known: POLISH, POLISH_TOL, POLISH_FAIL, POLISH_DEGREE, "
        "POLISH_MAX_ITER, POLISH_SPANS)")


def parse_indata_run_options(text: str) -> RunOptions:
    """Read every VMEX directive from raw ``&INDATA`` text.

    Both spellings are read; a repeated key must repeat the same value, and a
    conflicting repetition is an error rather than a silent last-one-wins.
    Ordinary Fortran comments and quoted ``!`` characters are unaffected: the
    directive patterns anchor on comment lines only, and VMEC2000 discards
    those lines entirely.
    """
    assignments: dict[str, Any] = {}

    def record(field: str, value: Any, *, key: str) -> None:
        if field in assignments and assignments[field] != value:
            raise VmecInputError(
                f"conflicting VMEX {key} directives: "
                f"{assignments[field]!r} then {value!r}")
        assignments[field] = value

    for key, token in _DIRECTIVE.findall(text):
        field, value = _parse_directive_value(key.upper(), token)
        record(field, value, key=key.upper())
    for token in _LEGACY_POLISH.findall(text):
        record("polish", _parse_polish(token, key="POLISH_FORCE_BALANCE"),
               key="POLISH_FORCE_BALANCE")
    return RunOptions(**assignments)


def strip_vmex_json(data: Mapping[str, Any]) -> tuple[dict[str, Any], RunOptions]:
    """Split a structured-JSON mapping into physics data and run options.

    The physics schema remains VMEC++ compatible after ``_vmex`` is removed.
    Unknown ``_vmex`` keys fail explicitly — a typo must not silently run
    with defaults.
    """
    physics = dict(data)
    section = physics.pop("_vmex", None)
    if section is None:
        return physics, RunOptions()
    if not isinstance(section, Mapping):
        raise VmecInputError("_vmex must be a JSON object")
    known = {field.name for field in fields(RunOptions)}
    unknown = sorted(set(section) - known)
    if unknown:
        raise VmecInputError(
            f"unknown _vmex keys {unknown} (known: {sorted(known)})")
    options = dict(section)
    if isinstance(options.get("polish"), str):
        options["polish"] = _parse_polish(options["polish"], key="polish")
    return physics, RunOptions(**options)


def format_indata_directives(options: RunOptions) -> str:
    """Serialize non-default options as canonical directive lines."""
    lines = []
    defaults = RunOptions()
    if options.polish != defaults.polish:
        token = "AUTO" if options.polish == "auto" else (
            ".TRUE." if options.polish else ".FALSE.")
        lines.append(f"!@VMEX POLISH = {token}")
    if options.polish_tol is not None:
        lines.append(f"!@VMEX POLISH_TOL = {options.polish_tol:.6E}")
    if options.polish_fail != defaults.polish_fail:
        lines.append(f"!@VMEX POLISH_FAIL = {options.polish_fail.upper()}")
    if options.polish_degree is not None:
        lines.append(f"!@VMEX POLISH_DEGREE = {options.polish_degree}")
    if options.polish_max_iter is not None:
        lines.append(f"!@VMEX POLISH_MAX_ITER = {options.polish_max_iter}")
    if options.polish_spans is not None:
        lines.append(f"!@VMEX POLISH_SPANS = {options.polish_spans}")
    if options.polish_budget is not None:
        lines.append(f"!@VMEX POLISH_BUDGET = {options.polish_budget:.6E}")
    return "\n".join(lines) + ("\n" if lines else "")


def read_input_request(path: str | Path) -> InputRequest:
    """Read one input file into physics plus execution options.

    ``VmecInput.from_file`` remains physics-only; this is the entry point the
    CLI and :func:`~vmex.core.multigrid.solve_file` share.
    """
    from .input import VmecInput

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json" or text.lstrip()[:1] == "{":
        import json

        # strip_vmex_json validates the section; from_json_text discards it
        # unvalidated so VmecInput.from_file keeps ignoring execution
        # metadata while preserving every physics key.
        _, options = strip_vmex_json(json.loads(text))
        inp = VmecInput.from_json_text(text)
    else:
        options = parse_indata_run_options(text)
        inp = VmecInput.from_indata_text(text)
    return InputRequest(input=inp, options=options, source=path)


def resolve_run_options(
    file_options: RunOptions | None,
    *,
    polish: bool | str | None = None,
    polish_tol: float | None = None,
    polish_fail: str | None = None,
    polish_degree: int | None = None,
    polish_max_iter: int | None = None,
    polish_spans: int | None = None,
    polish_budget: float | None = None,
) -> tuple[RunOptions, dict[str, str]]:
    """Apply the documented precedence and record where each value came from.

    Explicit Python keywords override the file; the CLI passes its flags
    through the same keywords, so ``CLI > Python > file > default`` reduces to
    one merge.  The source map (``"python"``, ``"file"``, ``"default"`` per
    field) goes into the run report so a surprising activation is traceable.
    """
    resolved = file_options if file_options is not None else RunOptions()
    source = {
        field.name: ("file" if file_options is not None
                     and getattr(file_options, field.name)
                     != getattr(RunOptions(), field.name) else "default")
        for field in fields(RunOptions)
    }
    overrides = {"polish": polish, "polish_tol": polish_tol,
                 "polish_fail": polish_fail, "polish_degree": polish_degree,
                 "polish_max_iter": polish_max_iter,
                 "polish_spans": polish_spans,
                 "polish_budget": polish_budget}
    updates = {name: value for name, value in overrides.items()
               if value is not None}
    if updates:
        resolved = replace(resolved, **updates)
        source.update({name: "python" for name in updates})
    return resolved, source


def polish_config_from_options(options: RunOptions, base: Any = None) -> Any:
    """Build the driver :class:`PolishConfig` implied by the run options.

    ``base`` is an explicit ``polish_config`` from the caller; explicit
    configuration wins over directive-level scalars, so options only fill a
    ``None`` base.  Returns ``None`` when nothing beyond driver defaults is
    requested, keeping the no-directive path bit-identical to before.
    """
    if base is not None:
        return base
    updates: dict[str, Any] = {}
    if options.polish_tol is not None:
        updates["tolerance"] = options.polish_tol
    if options.polish_degree is not None:
        updates["radial_degree"] = options.polish_degree
    if options.polish_max_iter is not None:
        updates["max_nonlinear_iterations"] = options.polish_max_iter
    if options.polish_spans is not None:
        updates["radial_spans"] = options.polish_spans
    if options.polish_budget is not None:
        updates["auto_budget_seconds"] = options.polish_budget
    if options.polish_fail in ("fallback", "warn"):
        updates["fail_policy"] = "return_unpolished"
    if not updates:
        return None
    from .polish_driver import PolishConfig

    return PolishConfig(**updates)
