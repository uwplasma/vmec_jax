#!/usr/bin/env python3
"""Docs prose and media gates (plan_docs_vmex.md sections 2 and 9).

Mechanical checks on the Diátaxis docs tree:

1.  Banned words/phrases in Markdown prose (fenced code and inline code are
    exempt), including exclamation marks and emoji in headings.
2.  Every how-to title starts with an imperative verb.
3.  Every page under ``docs/howto`` and ``docs/tutorials`` is <= 250 lines.
4.  No TODO/FIXME markers in any docs page.
5.  Media budget: files under ``docs/_static`` outside the frozen
    grandfather list (README-embedded figures and their committed input
    data) must total <= 500 KB with no single file over 150 KB.

Exit 0 when clean; exit 1 listing ``file:line: problem`` otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STATIC = DOCS / "_static"

# -- 1. banned patterns (plan section 2) ------------------------------------

BANNED = [
    r"\bdelve",
    r"\bleverag(e|es|ed|ing)\b",
    r"\butiliz(e|es|ed|ing)\b",
    r"\brobust(ly|ness)?\b",
    r"\bcutting[- ]edge\b",
    r"\bseamless(ly)?\b",
    r"\bpowerful\b",
    r"\bcomprehensive(ly)?\b",
    r"\bit'?s worth noting\b",
    r"\bin order to\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bbest[- ]in[- ]class\b",
]
BANNED_RE = [re.compile(p, re.IGNORECASE) for p in BANNED]

# Imperative first words allowed in how-to titles.
HOWTO_VERBS = {
    "run", "restart", "scan", "set", "scale", "optimize", "solve", "use",
    "diagnose", "plot", "troubleshoot", "build", "read", "write", "pick",
    "choose", "configure", "convert", "profile", "differentiate", "resume",
}

TODO_RE = re.compile(r"\b(TODO|FIXME|XXX)\b")
LINE_CAP = 250

# -- 5. media budget (bytes) -------------------------------------------------

BUDGET_TOTAL = 500 * 1024
BUDGET_FILE = 150 * 1024

# Frozen: files consumed by the top-level README.md (embedded figures) and
# the committed inputs of ``benchmarks/make_readme_figures.py`` /
# ``assets/manifest.json``. The README cannot move (its image paths are pinned
# by GitHub rendering and by tests), so these are exempt from the docs media
# budget — but the list itself must not grow: new media goes through the
# build-time or fetched channels of plan_docs_vmex.md section 8.
GRANDFATHERED_PREFIXES = (
    "qi_readme_cases/",
    "readme_best_cases/",
)
GRANDFATHERED_FILES = {
    "figures/freeb_diiid_mgrid_beta_ns101_panel_summary.csv",
    "figures/freeb_lpqa_direct_coil_beta_ns101_panel_summary.csv",
    "figures/minimal_seed_showcase_summary.csv",
    "figures/mirror_fixed_boundary_3d.png",
    "figures/mirror_free_boundary_beta50_summary.png",
    "figures/pr20_wout_parity_summary.json",
    "figures/qi_mirror_hybrid.png",
    "figures/readme_bootstrap.png",
    "figures/readme_convergence.png",
    "figures/readme_equilibrium_showcase.png",
    "figures/readme_essos_beta_scan.png",
    "figures/readme_objectives.png",
    "figures/readme_optimization.png",
    "figures/readme_optimization_landscapes.png",
    "figures/readme_precond.png",
    "figures/readme_qi.png",
    "figures/readme_runtime_compare.csv",
    "figures/readme_runtime_compare.json",
    "figures/readme_runtime_compare.png",
    "figures/readme_runtime_compare_current_vs_main.csv",
    "figures/readme_runtime_compare_current_vs_main.json",
    "figures/readme_runtime_compare_lpqa_rerun.csv",
    "figures/readme_runtime_compare_lpqa_rerun.json",
    "figures/readme_runtime_compare_regression_classifications.json",
    "figures/readme_single_stage.png",
    "figures/stellarator_mirror_hybrid.png",
}


def _is_grandfathered(rel: str) -> bool:
    return rel in GRANDFATHERED_FILES or rel.startswith(GRANDFATHERED_PREFIXES)


def _prose_lines(text: str):
    """Yield ``(lineno, line)`` for Markdown prose.

    Skips fenced code blocks and ``$$`` display-math blocks: code and LaTeX
    are not prose (``\\!`` spacing would otherwise read as an exclamation).
    """
    fence = None
    math = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        opener = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence is None and opener:
            fence = opener.group(1)[0] * 3
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped == "$$":
            math = not math
            continue
        if math:
            continue
        yield lineno, line


def _strip_inline_code(line: str) -> str:
    line = re.sub(r"`[^`]*`", "", line)
    return re.sub(r"\$[^$]*\$", "", line)


def _has_emoji(text: str) -> bool:
    return any(
        0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF
        for ch in text
    )


def check_markdown(path: Path, errors: list[str]) -> None:
    rel = path.relative_to(ROOT)
    text = path.read_text(encoding="utf-8")
    for lineno, line in _prose_lines(text):
        prose = _strip_inline_code(line)
        for pattern in BANNED_RE:
            if pattern.search(prose):
                errors.append(f"{rel}:{lineno}: banned phrase {pattern.pattern!r}")
        # Exclamation marks: image syntax ``![alt]`` is not an exclamation.
        if re.search(r"(?<!\w)!(?!\[)|\w!(?=[\s.,;:)\"']|$)", prose):
            errors.append(f"{rel}:{lineno}: exclamation mark in prose")
        if line.startswith("#") and _has_emoji(line):
            errors.append(f"{rel}:{lineno}: emoji in heading")


def check_howto_title(path: Path, errors: list[str]) -> None:
    rel = path.relative_to(ROOT)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            first = re.sub(r"[^A-Za-z].*", "", line[2:].strip().split()[0])
            if first.lower() not in HOWTO_VERBS:
                errors.append(
                    f"{rel}:1: how-to title must start with a verb "
                    f"(got {line[2:].strip()!r}; extend HOWTO_VERBS in "
                    "tools/check_docs_prose.py if the verb is legitimate)"
                )
            return
    errors.append(f"{rel}:1: no level-1 title found")


def check_media_budget(errors: list[str]) -> None:
    total = 0
    for path in sorted(STATIC.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        rel = path.relative_to(STATIC).as_posix()
        size = path.stat().st_size
        if _is_grandfathered(rel):
            continue
        total += size
        if size > BUDGET_FILE:
            errors.append(
                f"docs/_static/{rel}: {size // 1024} KB exceeds the "
                f"{BUDGET_FILE // 1024} KB per-file cap"
            )
    if total > BUDGET_TOTAL:
        errors.append(
            f"docs/_static: non-grandfathered media totals {total // 1024} KB "
            f"(cap {BUDGET_TOTAL // 1024} KB)"
        )


def main() -> int:
    errors: list[str] = []

    markdown_pages = sorted(DOCS.rglob("*.md"))
    markdown_pages = [p for p in markdown_pages if "_build" not in p.parts]
    for page in markdown_pages:
        check_markdown(page, errors)

    for page in markdown_pages:
        if page.parent == DOCS / "howto" and page.name != "index.md":
            check_howto_title(page, errors)

    for sub in ("howto", "tutorials"):
        for page in sorted((DOCS / sub).glob("*.md")):
            count = len(page.read_text(encoding="utf-8").splitlines())
            if count > LINE_CAP:
                errors.append(
                    f"{page.relative_to(ROOT)}: {count} lines exceeds the "
                    f"{LINE_CAP}-line cap for {sub} pages"
                )

    for page in sorted(DOCS.rglob("*")):
        if page.suffix in (".md", ".rst") and "_build" not in page.parts:
            for lineno, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
                if TODO_RE.search(line):
                    errors.append(f"{page.relative_to(ROOT)}:{lineno}: TODO marker")

    check_media_budget(errors)

    if errors:
        print("\n".join(errors))
        print(f"\n{len(errors)} docs-gate problem(s).")
        return 1
    print("docs prose and media gates clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
