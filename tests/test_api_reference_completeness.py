"""The API reference documents the whole public surface (plan.md 31.5 item 9).

Three guards, all cheap and import-light:

1. every public module of the ``vmex`` package appears in an ``automodule``
   directive somewhere under ``docs/`` (``MODULES_NOT_IN_THE_REFERENCE`` is the
   explicit, justified exception list);
2. every definition that ``automodule ... :members:`` renders from those
   modules carries a docstring (``UNDOCUMENTED`` is the allowlist, and it is
   the honest record of what is still missing);
3. every name in ``vmex.__all__`` resolves to an object with a docstring and is
   named in the top-level package docstring, which ``docs/reference/api/basic``
   presents as the index of the public API.

What "rendered by autodoc" means here, and why the checker is written on the
AST rather than on imported objects: ``:members:`` documents module-level
classes and functions and the public methods, properties, and classmethods of
public classes.  It does not document closures, methods of private classes, or
``@overload`` stubs, so neither does this test.  Working from the source also
keeps the check honest about *where* a definition lives: a name re-exported by
several modules is checked once, in the module that defines it.

``__init__`` bodies are excluded because ``autoclass_content`` is Sphinx's
default (``"class"``), so a constructor docstring is never rendered;
constructor arguments belong in the class docstring, where napoleon turns a
``Parameters`` section into the rendered signature documentation.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import vmex


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PACKAGE = ROOT / "vmex"

_AUTOMODULE = re.compile(r"^\s*\.\.\s+automodule::\s+([A-Za-z_][\w.]*)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Explicit exceptions.  Both lists are meant to shrink; never grow one to make
# a red test green without saying why in the comment next to the entry.
# ---------------------------------------------------------------------------

#: Public modules deliberately absent from the rendered reference.
MODULES_NOT_IN_THE_REFERENCE: frozenset[str] = frozenset()

#: ``module:qualname`` of public definitions that autodoc renders without a
#: docstring.  Empty is the goal state; an entry here is a documentation debt,
#: not a waiver of the contract.
UNDOCUMENTED: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _documented_modules() -> set[str]:
    """Every module named by an ``.. automodule::`` directive under ``docs/``."""
    found: set[str] = set()
    for path in sorted(DOCS.rglob("*")):
        if path.suffix not in (".rst", ".md") or "_build" in path.parts:
            continue
        found.update(_AUTOMODULE.findall(path.read_text(encoding="utf-8")))
    return found


def _public_modules() -> dict[str, Path]:
    """Importable modules of the package whose dotted path has no private part."""
    modules: dict[str, Path] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if any(part.startswith("_") for part in parts):
            continue
        modules[".".join(parts)] = path
    return modules


def _is_overload(node: ast.AST) -> bool:
    """Whether a def is an ``@overload`` stub (autodoc renders the real one)."""
    for decorator in getattr(node, "decorator_list", []):
        name = decorator
        if isinstance(name, ast.Call):
            name = name.func
        if isinstance(name, ast.Attribute):
            name = name.attr
        elif isinstance(name, ast.Name):
            name = name.id
        if name == "overload":
            return True
    return False


_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
#: Statements autodoc looks through: a def guarded by ``if``/``try`` is still
#: a module-level (or class-level) definition.
_TRANSPARENT = (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)


def _rendered_definitions(tree: ast.Module):
    """Yield ``(qualname, node)`` for what ``automodule ... :members:`` renders."""

    def walk(node: ast.AST, prefix: str, depth: int, in_public_class: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _TRANSPARENT):
                walk(child, prefix, depth, in_public_class)
                continue
            if not isinstance(child, _DEFINITION):
                continue
            qualname = f"{prefix}{child.name}"
            public = not child.name.startswith("_")
            rendered = public and (depth == 0 or (depth == 1 and in_public_class))
            if rendered and not _is_overload(child):
                yield qualname, child
            yield from walk(
                child,
                f"{qualname}.",
                depth + 1,
                isinstance(child, ast.ClassDef) and public,
            )

    yield from walk(tree, "", 0, False)


def _missing_docstrings() -> list[str]:
    """``module:qualname[:line]`` for every rendered definition with no docstring."""
    documented = _documented_modules()
    missing: list[str] = []
    for module, path in sorted(_public_modules().items()):
        if module not in documented:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(f"{module}:<module>:1")
        for qualname, node in _rendered_definitions(tree):
            if ast.get_docstring(node) is None:
                missing.append(f"{module}:{qualname}:{node.lineno}")
    return missing


def _key(entry: str) -> str:
    """Drop the line number so the allowlist survives unrelated edits."""
    module, qualname, _line = entry.rsplit(":", 2)
    return f"{module}:{qualname}"


# ---------------------------------------------------------------------------
# 1. Every public module is in the reference
# ---------------------------------------------------------------------------


def test_every_public_module_is_in_the_api_reference():
    absent = sorted(set(_public_modules()) - _documented_modules() - MODULES_NOT_IN_THE_REFERENCE)
    assert not absent, (
        "public modules with no '.. automodule::' entry under docs/: "
        + ", ".join(absent)
        + " — add one to docs/reference/api/basic.rst or advanced.rst, or list "
        "the module in MODULES_NOT_IN_THE_REFERENCE with the reason"
    )


def test_module_exception_list_is_not_stale():
    stale = sorted(MODULES_NOT_IN_THE_REFERENCE & _documented_modules())
    assert not stale, (
        "MODULES_NOT_IN_THE_REFERENCE names modules that are now documented: "
        + ", ".join(stale)
    )


# ---------------------------------------------------------------------------
# 2. Everything the reference renders has a docstring
# ---------------------------------------------------------------------------


def test_documented_modules_have_documented_public_definitions():
    missing = [entry for entry in _missing_docstrings() if _key(entry) not in UNDOCUMENTED]
    assert not missing, (
        f"{len(missing)} public definition(s) rendered by the API reference have "
        "no docstring:\n  " + "\n  ".join(missing)
    )


def test_docstring_allowlist_is_not_stale():
    still_missing = {_key(entry) for entry in _missing_docstrings()}
    stale = sorted(UNDOCUMENTED - still_missing)
    assert not stale, (
        "UNDOCUMENTED lists definitions that are now documented (or gone); "
        "remove them: " + ", ".join(stale)
    )


# ---------------------------------------------------------------------------
# 3. The exported surface
# ---------------------------------------------------------------------------


def _exports() -> list[str]:
    return [name for name in vmex.__all__ if name != "__version__"]


def test_every_export_has_a_docstring():
    # inspect.getdoc, not __doc__: autodoc_inherit_docstrings is on, so an
    # inherited docstring is what the reader actually sees.
    absent = [name for name in sorted(_exports()) if not (inspect.getdoc(getattr(vmex, name)) or "").strip()]
    assert not absent, (
        "names exported from vmex.__all__ and rendered in the API reference "
        "with no docstring: " + ", ".join(absent)
    )


def test_package_docstring_indexes_every_export():
    # docs/reference/api/basic.rst tells the reader this docstring lists every
    # name in vmex.__all__; keep that promise true.
    doc = vmex.__doc__ or ""
    absent = [name for name in _exports() if not re.search(rf"\b{re.escape(name)}\b", doc)]
    assert not absent, (
        "names exported from vmex.__all__ but missing from the package "
        "docstring index in vmex/__init__.py: " + ", ".join(sorted(absent))
    )
