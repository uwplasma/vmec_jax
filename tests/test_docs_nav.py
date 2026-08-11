"""The John test (plan_docs_vmex.md section 9.7).

Three questions a user actually asks must each be answerable in at most two
clicks from the docs landing page:

1. "How do I restart from a wout file?"  -> howto/restart-from-previous-run
2. "How do I run on GPU?"                -> howto/run-on-gpu
3. "Which VMEC2000 flags are supported?" -> reference/vmec2000-compatibility

Click 1 is any page linked from ``docs/index.md`` (inline links and toctree
entries); click 2 is any page linked from one of those. The primary check
parses the Markdown/rst sources so it needs no built site; when
``docs/_build/html`` exists, the same reachability is asserted on the built
HTML as well.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
HTML = DOCS / "_build" / "html"

TARGETS = {
    "How do I restart from a wout file?": "howto/restart-from-previous-run",
    "How do I run on GPU?": "howto/run-on-gpu",
    "Which VMEC2000 flags are supported?": "reference/vmec2000-compatibility",
}

_MD_LINK = re.compile(r"\]\(([^)#\s]+)\)")
_MYST_DOC_ROLE = re.compile(r"\{doc\}`(?:[^<`]*<)?([^>`]+)>?`")
_RST_DOC_ROLE = re.compile(r":doc:`(?:[^<`]*<)?([^>`]+)>?`")


def _normalize(base: Path, raw: str) -> str | None:
    """Resolve a link target to a docs-relative docname, or None."""
    raw = raw.strip()
    if raw.startswith(("http://", "https://", "mailto:")):
        return None
    for suffix in (".md", ".rst", ".html"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    if raw.startswith("/"):
        docname = raw.lstrip("/")
    else:
        docname = (base / raw).resolve().relative_to(DOCS.resolve()).as_posix()
    return docname


def _toctree_entries(text: str):
    """Toctree entry lines from MyST ```{toctree}``` and rst ``.. toctree::``."""
    for block in re.findall(r"```\{toctree\}(.*?)```", text, flags=re.DOTALL):
        for line in block.splitlines():
            entry = line.strip()
            if entry and not entry.startswith(":"):
                yield entry
    for block in re.findall(
        r"^\.\. toctree::\n((?:[ \t]+.*\n?)*)", text, flags=re.MULTILINE
    ):
        for line in block.splitlines():
            entry = line.strip()
            if entry and not entry.startswith(":"):
                yield entry


def _links_from_source(docname: str) -> set[str]:
    """Every docs page linked from one source page."""
    for suffix in (".md", ".rst"):
        path = DOCS / f"{docname}{suffix}"
        if path.exists():
            break
    else:
        return set()
    text = path.read_text(encoding="utf-8")
    base = path.parent
    found: set[str] = set()
    raw_targets = (
        [m.group(1) for m in _MD_LINK.finditer(text)]
        + [m.group(1) for m in _MYST_DOC_ROLE.finditer(text)]
        + [m.group(1) for m in _RST_DOC_ROLE.finditer(text)]
        + list(_toctree_entries(text))
    )
    for raw in raw_targets:
        try:
            docname_out = _normalize(base, raw)
        except ValueError:
            continue
        if docname_out:
            found.add(docname_out)
    return found


def _reachable_in_two_clicks_from_source() -> set[str]:
    click1 = _links_from_source("index")
    click2 = set(click1)
    for page in click1:
        click2 |= _links_from_source(page)
    return click2


@pytest.mark.parametrize("question,target", sorted(TARGETS.items()))
def test_john_question_within_two_clicks_of_landing(question, target):
    reachable = _reachable_in_two_clicks_from_source()
    assert target in reachable, (
        f"{question!r} -> docs page {target!r} is not reachable within two "
        f"clicks of docs/index.md; reachable set has {len(reachable)} pages"
    )


def _hrefs(html_path: Path) -> set[str]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    out = set()
    for href in re.findall(r'href="([^"#]+)"', text):
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        if href.endswith(".html"):
            try:
                resolved = (html_path.parent / href).resolve().relative_to(
                    HTML.resolve()
                )
            except ValueError:
                continue
            out.add(resolved.as_posix()[: -len(".html")])
    return out


@pytest.mark.skipif(
    not (HTML / "index.html").exists(),
    reason="docs are not built; the source-based John test covers navigation",
)
@pytest.mark.parametrize("question,target", sorted(TARGETS.items()))
def test_john_question_within_two_clicks_of_built_landing(question, target):
    click1 = _hrefs(HTML / "index.html")
    reachable = set(click1)
    for page in click1:
        page_html = HTML / f"{page}.html"
        if page_html.exists():
            reachable |= _hrefs(page_html)
    assert target in reachable, (
        f"{question!r} -> {target}.html is not linked from the built landing "
        "page or any page one click away"
    )
