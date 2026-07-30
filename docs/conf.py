from __future__ import annotations

import os
import sys
import tomllib
from datetime import date
from pathlib import Path


# -- Path setup ----------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# -- Project information --------------------------------------------------------

project = "VMEX"
author = "vmex contributors"
copyright = f"{date.today().year}, {author}"  # noqa: A001
with (_ROOT / "pyproject.toml").open("rb") as _f:
    release = tomllib.load(_f)["project"]["version"]
version = ".".join(release.split(".")[:2])

# Clean, un-versioned documentation title (browser tab / sidebar).
html_title = "VMEX documentation"


# -- General configuration ------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.duration",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no")


_ENABLE_VIEWCODE = _truthy(os.environ.get("SPHINX_VIEWCODE"))
if _ENABLE_VIEWCODE:
    extensions.append("sphinx.ext.viewcode")


_FAST = _truthy(os.environ.get("SPHINX_FAST"))
if _FAST:
    tags.add("fast")  # noqa: F821 - provided by the Sphinx configuration runtime
    # In fast mode build only a minimal landing page to keep CI under minutes.
    master_doc = "index_fast"
    include_patterns = ["index_fast.rst"]
    suppress_warnings = ["toc.not_readable", "toc.excluded"]

autosummary_generate = False
autosummary_imported_members = False
autosectionlabel_prefix_document = True
# Only label top-level page sections: section headings inside module
# docstrings (rendered by autodoc) would otherwise collide ("VMEC2000
# counterparts" appears in most vmex.core module docstrings).
autosectionlabel_maxdepth = 2
todo_include_todos = False

# Mock heavy runtime dependencies only when they are genuinely unavailable
# (e.g. a docs-only CI environment). With the real packages installed,
# autodoc imports vmex.core modules directly.
autodoc_mock_imports = []
for _mod in ("jax", "jaxlib", "netCDF4", "matplotlib", "scipy"):
    try:
        __import__(_mod)
    except Exception:
        autodoc_mock_imports.append(_mod)
autodoc_member_order = "bysource"

# sphinx-copybutton is an optional nicety; enable it when installed.
try:
    import sphinx_copybutton  # noqa: F401

    extensions.append("sphinx_copybutton")
    copybutton_prompt_text = r">>> |\.\.\. |\$ "
    copybutton_prompt_is_regexp = True
except Exception:
    pass

if _FAST:
    exclude_patterns += ["api/index.rst"]


# -- Options for HTML output ----------------------------------------------------

_theme = os.environ.get("SPHINX_THEME")
if _theme:
    html_theme = _theme
else:
    try:  # Prefer furo if installed (ReadTheDocs uses extras=[docs]).
        import furo  # noqa: F401

        html_theme = "furo"
    except Exception:
        # Keep local/offline builds working even if optional doc deps
        # (like furo) are not installed in the current environment.
        html_theme = "alabaster"
html_static_path = ["_static"]

if html_theme == "furo":
    html_theme_options = {
        "sidebar_hide_name": False,
        "light_css_variables": {
            "color-brand-primary": "#0f5c8c",
            "color-brand-content": "#0f5c8c",
        },
        "dark_css_variables": {
            "color-brand-primary": "#6fb7e6",
            "color-brand-content": "#6fb7e6",
        },
    }


# -- Intersphinx mapping --------------------------------------------------------

if os.environ.get("READTHEDOCS") == "True":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
else:
    # Offline/local builds in restricted environments (no network).
    intersphinx_mapping = {}
