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
