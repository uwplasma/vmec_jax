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
    "myst_parser",
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


# -- Options for MyST (Markdown) ------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3


# -- Options for HTML output ----------------------------------------------------

html_theme = os.environ.get("SPHINX_THEME", "furo")
html_static_path = ["_static"]


# -- Intersphinx mapping --------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable", {}),
}

