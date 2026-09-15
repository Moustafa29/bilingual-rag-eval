"""Near-duplicate chunks of each gold chunk.

UN documents repeat paragraphs word for word: preambles, standard requests, corrigenda that
reissue a whole document. If a question's answer sits in several near-identical chunks and only
one is labelled gold, every retriever is penalised for finding a copy. Each gold chunk's
relevance group therefore includes every chunk whose English 5-word shingles overlap it with
Jaccard similarity at or above the threshold. The chunks are aligned, so the same group applies
to Arabic.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from rageval.text import tokens


def shingles(text: str, size: int) -> set[str]:
    toks = tokens(text, "en")
    if len(toks) <= size:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + size]) for i in range(len(toks) - size + 1)}


def near_duplicate_groups(
    gold_ids: set[str], chunks: dict[str, dict], threshold: float, size: int = 5
) -> dict[str, list[str]]:
    gold_shingles = {g: shingles(chunks[g]["en"], size) for g in gold_ids}
    inverted: dict[str, set[str]] = defaultdict(set)
    for gold, grams in gold_shingles.items():
        for gram in grams:
            inverted[gram].add(gold)

    groups = {g: [g] for g in sorted(gold_ids)}
    for chunk_id, chunk in chunks.items():
        grams = shingles(chunk["en"], size)
        shared: Counter[str] = Counter()
        for gram in grams:
            for gold in inverted.get(gram, ()):
                shared[gold] += 1
        for gold, intersection in shared.items():
            if chunk_id == gold:
                continue
            union = len(grams) + len(gold_shingles[gold]) - intersection
            if union and intersection / union >= threshold:
                groups[gold].append(chunk_id)
    return groups
