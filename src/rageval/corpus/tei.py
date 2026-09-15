"""Parsing of UN Parallel Corpus v1.0 files as distributed by OPUS.

- `raw/<lang>/<doc_id>.xml`: the original TEI documents. Paragraph structure is kept
  (`<p id="31">`) and every sentence has an id `"<paragraph>:<sentence>"` (`<s id="31:2">`).
- `xml/ar-en.xml.gz`: the corpus's own sentence alignment, one `<linkGrp>` per document pair
  and one `<link>` per aligned group. `xtargets="34:1;31:1 31:2"` aligns Arabic sentence 34:1
  with English sentences 31:1 and 31:2. An empty side ("1-0", "0-1" links) marks a sentence
  with no counterpart in the other language.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

_SPACE = re.compile(r"\s+")
_GROUP = re.compile(r"<linkGrp\b([^>]*)>")
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_LINK = re.compile(r'<link\b[^>]*\bxtargets="([^"]*)"')


@dataclass(frozen=True)
class Sentence:
    sid: str
    pid: str
    text: str


@dataclass
class TeiDoc:
    symbol: str | None
    date: str | None
    sentences: list[Sentence]
    keywords: list[str] = field(default_factory=list)  # UNBIS subject terms from the header


@dataclass(frozen=True)
class Link:
    ar: tuple[str, ...]
    en: tuple[str, ...]

    @property
    def one_to_one(self) -> bool:
        return len(self.ar) == 1 and len(self.en) == 1


@dataclass
class LinkGroup:
    ar_path: str
    en_path: str
    score: float | None
    links: list[Link]


def parse_tei(data: bytes) -> TeiDoc:
    root = ET.fromstring(data)
    symbol = root.find(".//idno[@type='symbol']")
    date = root.find(".//publicationStmt/date")
    sentences = []
    body = root.find(".//body")
    if body is not None:
        for p in body.iter("p"):
            pid = p.get("id", "")
            for s in p.iter("s"):
                text = _SPACE.sub(" ", "".join(s.itertext())).strip()
                sentences.append(Sentence(s.get("id", ""), pid, text))
    return TeiDoc(
        symbol=symbol.text.strip() if symbol is not None and symbol.text else None,
        date=date.text.strip() if date is not None and date.text else None,
        sentences=sentences,
        keywords=[t.text.strip() for t in root.findall(".//keywords/term") if t.text and t.text.strip()],
    )


def parse_xtargets(value: str) -> Link:
    ar, en = value.split(";")
    return Link(ar=tuple(ar.split()), en=tuple(en.split()))


def doc_id_from_path(path: str, lang: str) -> str:
    """`"ar/1992/a/cn_4/452.xml.gz"` -> `"1992/a/cn_4/452"`."""
    prefix = f"{lang}/"
    if not path.startswith(prefix):
        raise ValueError(f"expected a path under {prefix!r}: {path}")
    doc = path[len(prefix) :]
    for suffix in (".xml.gz", ".xml"):
        if doc.endswith(suffix):
            return doc[: -len(suffix)]
    return doc


def iter_link_groups(
    lines: Iterable[str], keep: Callable[[str, str], bool] | None = None
) -> Iterator[LinkGroup]:
    """Stream link groups from the alignment file, one element per line as OPUS writes it.

    `keep(ar_path, en_path)` skips unwanted groups without parsing their links, so extracting a
    few thousand documents does not allocate objects for the other ~110,000.
    """
    group = None
    skipping = False
    for line in lines:
        if skipping:
            if "</linkGrp>" in line:
                skipping = False
            continue
        match = _LINK.search(line)
        if match and group is not None:
            group.links.append(parse_xtargets(match.group(1)))
            continue
        match = _GROUP.search(line)
        if match:
            attrs = dict(_ATTR.findall(match.group(1)))
            if keep is not None and not keep(attrs["fromDoc"], attrs["toDoc"]):
                group, skipping = None, True
                continue
            score = attrs.get("score")
            group = LinkGroup(attrs["fromDoc"], attrs["toDoc"], float(score) if score else None, [])
            continue
        if group is not None and "</linkGrp>" in line:
            yield group
            group = None
