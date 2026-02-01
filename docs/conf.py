from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path


# -- Path setup ----------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# -- Project information --------------------------------------------------------

project = "vmec-jax"
author = "vmec_jax contributors"
copyright = f"{date.today().year}, {author}"  # noqa: A001


# -- General configuration ------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.duration",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autosectionlabel_prefix_document = True
todo_include_todos = False


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


# -- Intersphinx mapping --------------------------------------------------------

if os.environ.get("READTHEDOCS") == "True":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
else:
    # Offline/local builds in restricted environments (no network).
    intersphinx_mapping = {}
