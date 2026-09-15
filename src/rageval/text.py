"""Text normalization and matching shared by question filters, answer scoring and BM25.

Arabic normalization follows Lucene's ArabicNormalizer (the `arabic_normalization` step of
Elasticsearch's `arabic` analyzer), with two additions:

- NFKC first. Older UN Arabic files contain Arabic Presentation Forms, e.g. the lam-alef
  ligature U+FEF7, left over from legacy word-processor conversion. They look identical on
  screen but never match the base letters unless folded.
- Arabic-Indic digits are folded to ASCII, so "٤٧/٣٣" matches "47/33".
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_TASHKEEL = re.compile("[ً-ٰٟ]")
_TATWEEL = "ـ"
_AR_LETTERS = str.maketrans(
    {
        "آ": "ا",  # alef with madda -> alef
        "أ": "ا",  # alef with hamza above -> alef
        "إ": "ا",  # alef with hamza below -> alef
        "ٱ": "ا",  # alef wasla -> alef
        "ة": "ه",  # teh marbuta -> heh
        "ى": "ي",  # alef maksura -> yeh
    }
)
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)
_WORD = re.compile(r"\w+")
_SPACE = re.compile(r"\s+")

# Longest first, so "وال" is stripped before "ال".
_AR_DEFINITE_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_EN_ARTICLES = frozenset({"a", "an", "the"})

LANGS = ("en", "ar")


def normalize(text: str, lang: str) -> str:
    if lang not in LANGS:
        raise ValueError(f"unsupported language: {lang}")
    text = unicodedata.normalize("NFKC", text).translate(_DIGITS).lower()
    if lang == "ar":
        text = _TASHKEEL.sub("", text).replace(_TATWEEL, "").translate(_AR_LETTERS)
    return _SPACE.sub(" ", text).strip()


def tokens(text: str, lang: str) -> list[str]:
    return _WORD.findall(normalize(text, lang))


def answer_tokens(text: str, lang: str) -> list[str]:
    """Tokens for comparing short answers: drops English articles and the Arabic definite article."""
    out = []
    for tok in tokens(text, lang):
        if lang == "en":
            if tok not in _EN_ARTICLES:
                out.append(tok)
            continue
        for prefix in _AR_DEFINITE_PREFIXES:
            if tok.startswith(prefix) and len(tok) - len(prefix) >= 2:
                tok = tok[len(prefix) :]
                break
        out.append(tok)
    return out


def token_f1(prediction: str, reference: str, lang: str) -> float:
    """SQuAD-style token F1 on normalized answer tokens."""
    pred = answer_tokens(prediction, lang)
    ref = answer_tokens(reference, lang)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_span(haystack: str, span: str, lang: str) -> bool:
    """True if `span` occurs in `haystack` as a whole-token sequence, after normalization."""
    needle = tokens(span, lang)
    if not needle:
        return False
    return f" {' '.join(needle)} " in f" {' '.join(tokens(haystack, lang))} "


_AR_CONJUNCTIONS = ("و", "ف")


def _strip_conjunction(token: str) -> str:
    return token[1:] if len(token) >= 3 and token[0] in _AR_CONJUNCTIONS else token


def contains_evidence(haystack: str, span: str, lang: str) -> bool:
    """Containment for copied evidence sentences, tolerant of two copy variants seen in the pilot.

    - Arabic: a dropped or added conjunction clitic ("وإذ يؤكد" copied as "إذ يؤكد"). One leading و or
      ف is removed from every token on both sides before comparing.
    - English: hyphens the corpus text lost ("SecretaryGeneral's" in the passage, "Secretary-General's"
      in the copy). Token sequences are compared with spaces removed, for spans of 5+ tokens only.

    Evidence is a whole sentence, so this looseness cannot turn a different sentence into a match.
    Short answers keep the strict `contains_span`.
    """
    if contains_span(haystack, span, lang):
        return True
    hay, needle = tokens(haystack, lang), tokens(span, lang)
    if not needle:
        return False
    if lang == "ar":
        hay, needle = [_strip_conjunction(t) for t in hay], [_strip_conjunction(t) for t in needle]
        return f" {' '.join(needle)} " in f" {' '.join(hay)} "
    return len(needle) >= 5 and "".join(needle) in "".join(hay)
