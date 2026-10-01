"""Turning an arbitrary uploaded PDF into the same units everything else works on.

The built-in corpora know their own structure, so their adapters can be precise. An
uploaded PDF could be anything, so this one works by degrees: look for numbered headings
first, because a contract or a standard almost always has them and they make far better
boundaries than a character count; fall back to splitting on size only when there is no
structure to find.
"""
import re
from pathlib import Path

import pymupdf

from anchor.corpora import Unit, tidy

# "7.", "7.2", "7.2.1", optionally preceded by a word like Section or Clause, at the start
# of a line and followed by a title rather than a sentence.
HEADING = re.compile(
    r"^\s*(?:(?:section|clause|article|chapter|part|annex|appendix|schedule)\s+)?"
    r"([0-9]+(?:\.[0-9]+){0,3}|[IVXLC]+)[.):]?\s+([A-Z][^\n]{2,80})$",
    re.IGNORECASE | re.MULTILINE,
)

# What a reference to another part of the same document looks like.
REFERENCE = re.compile(
    r"\b(?:section|clause|article|chapter|part|annex|appendix|schedule|paragraph|para)\s+"
    r"([0-9]+(?:\.[0-9]+){0,3}|[IVXLC]+)\b",
    re.IGNORECASE,
)

PAGE_NOISE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE)


def read(path: str | Path) -> tuple[str, int]:
    """All the text, plus the page count, with the most obvious furniture removed."""
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        text = PAGE_NOISE.sub("", text)
        pages.append(text)
    count = len(doc)
    doc.close()
    body = "\n\n".join(pages)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip(), count


def refs_in(text: str) -> list[str]:
    return sorted({m.group(1).lower() for m in REFERENCE.finditer(text)})


def by_heading(text: str, label: str, url: str) -> list[Unit]:
    marks = list(HEADING.finditer(text))
    if len(marks) < 3:
        return []

    units = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = tidy(text[start:end])
        if len(body) < 60:
            continue
        number, title = m.group(1), tidy(m.group(2))
        units.append(Unit(
            unit_id=number.lower(), label=f"Section {number}", title=title,
            part="", text=f"{title}. {body}"[:4000], url=url,
            refs=tuple(r for r in refs_in(body) if r != number.lower()),
        ))
    return units


def by_size(text: str, label: str, url: str, size: int = 1100, overlap: int = 200) -> list[Unit]:
    units = []
    step = size - overlap
    for i in range(0, len(text), step):
        window = tidy(text[i:i + size])
        if len(window) < 120:
            continue
        n = len(units) + 1
        units.append(Unit(
            unit_id=f"part-{n}", label=f"Part {n}", title=label, part="",
            text=window, url=url, refs=tuple(refs_in(window)),
        ))
    return units


def load(path: str | Path, name: str) -> tuple[list[Unit], dict]:
    """Units for an uploaded PDF, plus what was found, so a caller can report it."""
    text, pages = read(path)
    url = f"upload://{name}"

    units = by_heading(text, name, url)
    how = "numbered sections"
    if not units:
        units = by_size(text, name, url)
        how = "fixed-size chunks (no numbered headings found)"

    edges = sum(len(u.refs) for u in units)
    return units, {
        "pages": pages,
        "characters": len(text),
        "units": len(units),
        "edges": edges,
        "split_by": how,
    }
