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
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.rediraffe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.duration",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no")


if _truthy(os.environ.get("SPHINX_VIEWCODE")):
    extensions.append("sphinx.ext.viewcode")

autosummary_generate = False
autosummary_imported_members = False
autosectionlabel_prefix_document = True
# Only label top-level page sections: section headings inside module
# docstrings (rendered by autodoc) would otherwise collide ("VMEC2000
# counterparts" appears in most vmex.core module docstrings).
autosectionlabel_maxdepth = 2

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

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# -- Redirects (old flat tree -> Diátaxis tree) ---------------------------------
# One entry per pre-restructure page, so bookmarks and the README's pinned
# https://vmex.readthedocs.io/en/latest/capabilities.html link keep working.

rediraffe_redirects = {
    "quickstart": "all-of-vmex",
    "tutorials": "tutorials/index",
    "cli": "reference/cli",
    "input_reference": "reference/input-file",
    "wout_reference": "reference/wout-file",
    "objectives": "reference/objectives",
    "optimization": "reference/optimization",
    "vmec2000_compatibility": "reference/vmec2000-compatibility",
    "capabilities": "reference/capabilities",
    "performance": "reference/performance",
    "equations": "explanation/variational-problem",
    "theory": "explanation/spectral-representation",
    "algorithms": "explanation/iteration",
    "architecture": "explanation/architecture",
    "confinement": "explanation/confinement",
    "mirror_geometry": "explanation/mirror-geometry",
    "parallelization": "explanation/parallelization",
    "scaling": "howto/scale-a-configuration",
    "contributing": "project/contributing",
    "references": "project/references",
    "api/index": "reference/api/basic",
}


# -- Options for HTML output ----------------------------------------------------

_theme = os.environ.get("SPHINX_THEME")
if _theme:
    html_theme = _theme
else:
    html_theme = "furo"
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


# -- Linkcheck ------------------------------------------------------------------
# Hosts that rate-limit or require auth from CI runners; every entry states
# its reason. The weekly linkcheck job builds with `-b linkcheck`.

linkcheck_ignore = [
    r"https://doi\.org/.*",          # DOI redirects intermittently 403 robots
    r"https://meetings\.aps\.org/.*",  # APS blocks non-browser agents
    r"https://downloads\.regulations\.gov/.*",  # S3 signed-URL host, 403 to bots
    # Generated evidence links into this repository 404 on a PR whose commit
    # is ahead of main; file existence is enforced locally by
    # tests/test_capability_docs.py, which is stronger than a URL probe.
    r"https://github\.com/uwplasma/VMEX/blob/main/.*",
]
linkcheck_timeout = 30
# github.com aborts a share of a runner's concurrent probes outright
# ("RemoteDisconnected") rather than answering 429, so
# linkcheck_rate_limit_timeout never engages. Fewer parallel probes and a retry
# clear that without hiding a genuinely dead link.
linkcheck_workers = 2
linkcheck_retries = 3
