"""Tokenizers for BM25, in three strengths, so the Arabic lexical baseline is not a straw man.

- `raw`:   lowercase + split on non-word characters. What a naive BM25 does.
- `norm`:  + the normalization in `rageval.text.normalize` (Arabic: NFKC, diacritics, alef
           variants, teh marbuta, alef maksura, tatweel; both: Arabic-Indic digits).
- `light`: + light stemming. English uses the Snowball (Porter2) stemmer. Arabic follows
           Lucene's `ArabicStemmer`, the stemmer behind Elasticsearch's `arabic` analyzer,
           itself based on Larkey, Ballesteros & Connell's light stemming: strip one common
           prefix (the definite article and conjunction/preposition clitics), then common
           suffixes. It does not undo broken plurals (كتاب -> كتب): those change the word's
           internal vowel pattern, not its affixes.

Whitespace tokenization of Arabic leaves clitics attached, so "والمنظمة" (and the organization)
never matches "منظمة" (organization). The gap between `raw` and `light` BM25 in Arabic measures
how much of the lexical baseline's Arabic penalty is the tokenizer.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

from rageval.text import normalize

ANALYZERS = ("raw", "norm", "light")

_WORD = re.compile(r"\w+")

# Lucene ArabicStemmer, applied after normalization. Prefix order matters: the first match wins.
_AR_PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل", "و")
# Every matching suffix is stripped in this order (Lucene loops over all of them once).
# "ية" and "ة" are listed for fidelity; after normalization teh marbuta is already heh.
_AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "يه", "ية", "ه", "ة", "ي")

_english = snowballstemmer.stemmer("english")


def stem_arabic(word: str) -> str:
    for prefix in _AR_PREFIXES:
        # A one-letter prefix (wa-) needs at least 4 letters in the word; others leave 2.
        min_len = 4 if len(prefix) == 1 else len(prefix) + 2
        if len(word) >= min_len and word.startswith(prefix):
            word = word[len(prefix) :]
            break
    for suffix in _AR_SUFFIXES:
        if len(word) >= len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
    return word


@lru_cache(maxsize=500_000)
def _stem_english(word: str) -> str:
    return _english.stemWord(word)


def analyze(text: str, lang: str, analyzer: str) -> list[str]:
    if analyzer == "raw":
        return _WORD.findall(unicodedata.normalize("NFC", text).lower())
    tokens = _WORD.findall(normalize(text, lang))
    if analyzer == "norm":
        return tokens
    if analyzer == "light":
        stem = stem_arabic if lang == "ar" else _stem_english
        return [stem(t) for t in tokens]
    raise ValueError(f"unknown analyzer: {analyzer}")
