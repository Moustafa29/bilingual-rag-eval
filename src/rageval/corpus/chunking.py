"""Aligned chunking: chunk k of a document holds the same content in English and Arabic.

Chunks are built from *beads*. A bead is one alignment link: the Arabic sentence(s) and the
English sentence(s) the corpus aligns to each other ("1-1", "1-2", ...), or a sentence with no
counterpart ("1-0", "0-1"). Consecutive beads are packed into a chunk. A chunk closes once it
has reached the target size at the end of a paragraph, and never grows past the maximum size in
either language. Boundaries always fall between beads, so both language versions of a chunk
cover the same aligned sentences and share one chunk id.

Sizes are counted in the embedding model's subword tokens, per language, and the tighter
language decides. The XLM-R tokenizer needs more tokens for Arabic than for the same English
sentence, so sizing on English alone would let Arabic chunks overflow the model's 512-token
limit and be truncated silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rageval.corpus.tei import Link, Sentence, TeiDoc


class AlignmentError(ValueError):
    pass


@dataclass
class Bead:
    en: list[Sentence]
    ar: list[Sentence]
    paragraph_end: bool

    @property
    def one_to_one(self) -> bool:
        return len(self.en) == 1 and len(self.ar) == 1


def _last_in_paragraph(sentences: Sequence[Sentence]) -> set[str]:
    return {
        s.sid
        for i, s in enumerate(sentences)
        if i + 1 == len(sentences) or sentences[i + 1].pid != s.pid
    }


def _positions(ids: tuple[str, ...], index: dict[str, int], lang: str) -> list[int]:
    try:
        return [index[sid] for sid in ids]
    except KeyError as missing:
        raise AlignmentError(f"link references missing {lang} sentence {missing}") from None


def build_beads(en_doc: TeiDoc, ar_doc: TeiDoc, links: Sequence[Link]) -> list[Bead]:
    en_index = {s.sid: i for i, s in enumerate(en_doc.sentences)}
    ar_index = {s.sid: i for i, s in enumerate(ar_doc.sentences)}
    en_ends = _last_in_paragraph(en_doc.sentences)
    ar_ends = _last_in_paragraph(ar_doc.sentences)
    last = {"en": -1, "ar": -1}
    beads = []
    for link in links:
        en_pos = _positions(link.en, en_index, "en")
        ar_pos = _positions(link.ar, ar_index, "ar")
        for lang, pos in (("en", en_pos), ("ar", ar_pos)):
            if pos and (pos != sorted(pos) or pos[0] <= last[lang]):
                raise AlignmentError(f"non-monotonic {lang} alignment at {link}")
            if pos:
                last[lang] = pos[-1]
        en = [en_doc.sentences[i] for i in en_pos]
        ar = [ar_doc.sentences[i] for i in ar_pos]
        if not en and not ar:
            continue
        end = en[-1].sid in en_ends if en else ar[-1].sid in ar_ends
        beads.append(Bead(en=en, ar=ar, paragraph_end=end))
    return beads


def join_sentences(sentences: Sequence[Sentence]) -> str:
    """Sentences in one paragraph are joined by a space, paragraphs by a newline."""
    parts: list[str] = []
    previous_pid = None
    for s in sentences:
        if not s.text:
            continue
        if previous_pid is not None:
            parts.append(" " if s.pid == previous_pid else "\n")
        parts.append(s.text)
        previous_pid = s.pid
    return "".join(parts)


def pack_beads(
    paragraph_ends: Sequence[bool],
    en_counts: Sequence[int],
    ar_counts: Sequence[int],
    target_tokens: int,
    max_tokens: int,
) -> list[list[int]]:
    """Group bead indices into chunks. A single bead larger than `max_tokens` becomes its own chunk."""
    groups: list[list[int]] = []
    current: list[int] = []
    en_total = ar_total = 0
    for i, paragraph_end in enumerate(paragraph_ends):
        if current and (en_total + en_counts[i] > max_tokens or ar_total + ar_counts[i] > max_tokens):
            groups.append(current)
            current, en_total, ar_total = [], 0, 0
        current.append(i)
        en_total += en_counts[i]
        ar_total += ar_counts[i]
        if paragraph_end and max(en_total, ar_total) >= target_tokens:
            groups.append(current)
            current, en_total, ar_total = [], 0, 0
    if current:
        groups.append(current)
    return groups


def make_chunks(
    doc_id: str,
    beads: Sequence[Bead],
    count_tokens: Callable[[list[str]], list[int]],
    target_tokens: int,
    max_tokens: int,
) -> list[dict]:
    en_texts = [join_sentences(b.en) for b in beads]
    ar_texts = [join_sentences(b.ar) for b in beads]
    groups = pack_beads(
        [b.paragraph_end for b in beads],
        count_tokens(en_texts),
        count_tokens(ar_texts),
        target_tokens,
        max_tokens,
    )
    chunks = []
    for k, group in enumerate(groups):
        members = [beads[i] for i in group]
        chunks.append(
            {
                "chunk_id": f"{doc_id}#{k:04d}",
                "doc_id": doc_id,
                "en": join_sentences([s for b in members for s in b.en]),
                "ar": join_sentences([s for b in members for s in b.ar]),
                "n_beads": len(members),
                "one_to_one_ratio": sum(b.one_to_one for b in members) / len(members),
            }
        )
    return chunks


class TokenCounter:
    """Counts subword tokens with the embedding model's own tokenizer.

    Uses the `tokenizers` library directly on the model's tokenizer.json rather than
    `transformers`, which imports far more and ran a 7.7 GB laptop out of memory mid-build.
    """

    def __init__(self, model_name: str):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_pretrained(model_name)
        self.tokenizer.no_truncation()

    def __call__(self, texts: list[str], prefix: str = "", special_tokens: bool = False) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer.encode_batch([prefix + t for t in texts], add_special_tokens=special_tokens)
        return [len(e.ids) for e in encoded]
