import json
import re
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

from anchor.fetch import get

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
         "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13}


@dataclass
class Unit:
    unit_id: str
    label: str
    title: str
    part: str
    text: str
    url: str
    refs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def citation(self) -> str:
        return f"{self.label}({self.part})" if self.part else self.label


def tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


class Corpus:
    """What a document set has to tell Anchor about itself.

    Everything downstream of this is corpus blind, so adding a new source means
    answering three questions here: where the documents live, how one splits into
    units, and what a reference to another unit looks like in the text.
    """

    name = ""

    def sources(self) -> list[str]:
        raise NotImplementedError

    def units(self, source: str) -> list[Unit]:
        raise NotImplementedError

    def refs_in(self, text: str) -> list[str]:
        raise NotImplementedError


class AIAct(Corpus):
    """Regulation (EU) 2024/1689, read from the articles and annexes.

    Recitals are left out on purpose. They explain the reasoning behind the law but
    are not the law, and including them roughly triples the corpus with text that
    sounds authoritative without being binding, which is exactly the kind of thing
    you do not want a grounded system quoting back as a rule.
    """

    name = "aiact"
    base = "https://artificialintelligenceact.eu"

    def sources(self) -> list[str]:
        return [f"article-{n}" for n in range(1, 114)] + [f"annex-{n}" for n in range(1, 14)]

    def url_for(self, source: str) -> str:
        kind, number = source.split("-")
        return f"{self.base}/{kind}/{number}/"

    def units(self, source: str) -> list[Unit]:
        url = self.url_for(source)
        try:
            html = get(url)
        except httpx.HTTPStatusError:
            return []

        soup = BeautifulSoup(html, "lxml")

        # These are hover definitions the site injects into the body text. Left in,
        # every mention of "AI system" drags its whole legal definition along with it
        # and the chunk stops being about what the article actually says.
        for tip in soup.select(".term-tip"):
            tip.decompose()

        raw_title = soup.select_one("title")
        title = raw_title.get_text(" ", strip=True).split("|")[0].strip() if raw_title else source
        label = title.split(":")[0].strip()

        groups: list[tuple[str, list[str]]] = []
        for para in soup.select(".para"):
            label_node = para.select_one(".para-label")
            text_node = para.select_one(".para-text")
            if text_node is None:
                continue

            mark = tidy(label_node.get_text(" ", strip=True)) if label_node else ""
            body = tidy(text_node.get_text(" ", strip=True))
            if not body:
                continue

            # Points like (a) and (b) are continuations of the numbered paragraph above
            # them. On their own they are unreadable, so they ride along with their parent.
            if groups and (mark.startswith("(") or not mark):
                groups[-1][1].append(f"{mark} {body}".strip())
            else:
                groups.append((mark.rstrip("."), [body]))

        # An unlabelled opening line ending in a colon is introducing the list below it,
        # not saying anything itself. Left as its own chunk it wins searches it cannot
        # answer, because it is the piece that repeats the words of the question while
        # the actual provisions underneath are phrased in their own terms. Annex III is
        # the clear case: its lead-in scored higher for a question about screening job
        # applicants than the employment entry that answers it. So it rides on the front
        # of each item instead, which also makes every item readable on its own.
        lead = ""
        if len(groups) > 1 and not groups[0][0] and groups[0][1][-1].rstrip().endswith(":"):
            lead = " ".join(groups[0][1])
            groups = groups[1:]

        units = []
        for part, lines in groups:
            text = " ".join(lines)
            if lead:
                text = f"{lead} {text}"
            units.append(Unit(
                unit_id=source,
                label=label,
                title=title,
                part=part,
                text=text,
                url=url,
                refs=tuple(self.refs_in(text)),
            ))
        return units

    def refs_in(self, text: str) -> list[str]:
        found = []
        for number in re.findall(r"\bArticles?\s+(\d{1,3})", text):
            found.append(f"article-{number}")
        for numeral in re.findall(r"\bAnnex(?:es)?\s+([IVX]+)\b", text):
            if numeral in ROMAN:
                found.append(f"annex-{ROMAN[numeral]}")
        return sorted(set(found))


class PEPs(Corpus):
    """Python Enhancement Proposals, used to check the engine is not quietly
    specialised to legal text. Different writing, different structure, same job."""

    name = "peps"
    base = "https://peps.python.org"

    def __init__(self, limit: int = 60):
        self.limit = limit

    def sources(self) -> list[str]:
        index = json.loads(get(f"{self.base}/api/peps.json"))
        keep = [
            int(number) for number, meta in index.items()
            if meta.get("status") in ("Active", "Final")
            and meta.get("type") in ("Process", "Informational")
            and int(number) > 0
        ]
        return [f"pep-{n}" for n in sorted(keep)[: self.limit]]

    def url_for(self, source: str) -> str:
        return f"{self.base}/pep-{int(source.split('-')[1]):04d}/"

    def units(self, source: str) -> list[Unit]:
        url = self.url_for(source)
        try:
            html = get(url)
        except httpx.HTTPStatusError:
            return []

        soup = BeautifulSoup(html, "lxml")
        number = source.split("-")[1]
        heading = soup.select_one("h1")
        title = tidy(heading.get_text(" ", strip=True)) if heading else source
        label = f"PEP {number}"

        units = []
        for section in soup.select("section"):
            head = section.find(["h1", "h2", "h3"], recursive=False)
            if head is None:
                continue

            part = tidy(head.get_text(" ", strip=True)).rstrip("¶")

            # Only the prose directly under this heading. Without this, a top level
            # section swallows every subsection under it and the same sentences end up
            # embedded three times at three different sizes.
            body = []
            for child in section.find_all(["p", "li", "pre"], recursive=False):
                body.append(tidy(child.get_text(" ", strip=True)))
            for block in section.find_all(["ul", "ol", "dl"], recursive=False):
                body.append(tidy(block.get_text(" ", strip=True)))

            text = " ".join(b for b in body if b)
            if len(text) < 80:
                continue

            units.append(Unit(
                unit_id=source,
                label=label,
                title=title,
                part=part,
                text=text,
                url=url,
                refs=tuple(r for r in self.refs_in(text) if r != source),
            ))
        return units

    def refs_in(self, text: str) -> list[str]:
        numbers = re.findall(r"\bPEP\s*#?\s*(\d{1,4})\b", text)
        return sorted({f"pep-{int(n)}" for n in numbers})


REGISTRY = {"aiact": AIAct, "peps": PEPs}
