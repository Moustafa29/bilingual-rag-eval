"""LLM question generation with verification. Every rejection is recorded with its reason.

Single-hop, per candidate chunk:
  1. generate  (generator, source language, source passage): question, short answer, evidence
     sentence. Rule checks: evidence and answer occur in the passage; no "this document" phrasing.
  2. translate (generator, *no passage*): question and answer into the target language. The
     passage is withheld so the translation cannot borrow its wording, which would inflate
     lexical overlap and favour BM25 in the target language.
  3. verify    (verifier, a different model, target language, target passage): answer the
     translated question from the passage alone. The span must occur in the passage and match
     the translated answer. The verified span becomes the target-language reference answer,
     because it is the passage's official wording.

Bridge (two-hop), per candidate (A cites document B):
  1. generate from both source passages; the answer must be in B and the question must not
     contain a document symbol or resolution number.
  2. translate, as above.
  3. shortcut checks in English: the verifier must NOT answer correctly from B alone or from A
     alone. If either passage suffices, the question is not really two-hop.
  4. verify from both target passages; the answer must come from B.

Direction alternates en->ar / ar->en so that neither language always receives the machine-
translated question. Once one direction's quota is full, the other is used.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from rageval.llm.client import DailyLimitReached
from rageval.questions.checks import has_context_reference, is_true, mentions_document_number, parse_json_object
from rageval.text import contains_span, token_f1

LANG_NAMES = {"en": "English", "ar": "Arabic"}
DIRECTIONS = ("en->ar", "ar->en")


class Prompts:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def render(self, name: str, **values: str) -> str:
        template = (self.directory / f"{name}.txt").read_text(encoding="utf-8")
        return Template(template).substitute(values)


def format_passages(passages: list[str]) -> str:
    return "\n\n".join(f"Passage {i}:\n<<<\n{p}\n>>>" for i, p in enumerate(passages, 1))


@dataclass
class Attempt:
    status: str  # "accepted" or the rejection reason
    record: dict | None
    calls: list[str] = field(default_factory=list)
    usage: Counter = field(default_factory=Counter)


class _Rejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class QuestionBuilder:
    def __init__(self, generator, verifier, prompts: Prompts, f1_threshold: float):
        self.generator = generator
        self.verifier = verifier
        self.prompts = prompts
        self.f1_threshold = f1_threshold

    def _ask(self, client, prompt: str, attempt: Attempt) -> dict:
        result = client.chat([{"role": "user", "content": prompt}])
        attempt.calls.append(result.cache_key)
        if not result.cached:
            attempt.usage[f"{client.model}:prompt"] += result.usage.get("prompt_tokens", 0)
            attempt.usage[f"{client.model}:completion"] += result.usage.get("completion_tokens", 0)
        parsed = parse_json_object(result.text)
        if parsed is None:
            raise _Rejected("bad_json")
        return parsed

    def _verify(self, lang: str, question: str, passages: list[str], attempt: Attempt) -> str | None:
        prompt = self.prompts.render(
            "verify", lang_name=LANG_NAMES[lang], question=question, passages=format_passages(passages)
        )
        answer = self._ask(self.verifier, prompt, attempt)
        return str(answer.get("answer", "")).strip() if is_true(answer.get("answerable")) else None

    def _translate(self, src: str, tgt: str, question: str, answer: str, attempt: Attempt) -> tuple[str, str]:
        prompt = self.prompts.render(
            "translate",
            src_lang_name=LANG_NAMES[src],
            tgt_lang_name=LANG_NAMES[tgt],
            question=question,
            answer=answer,
        )
        out = self._ask(self.generator, prompt, attempt)
        q, a = str(out.get("question", "")).strip(), str(out.get("answer", "")).strip()
        if not q or not a:
            raise _Rejected("translation_missing_fields")
        if has_context_reference(q, tgt):
            raise _Rejected("context_reference_translated")
        return q, a

    def single(self, chunk: dict, direction: str) -> Attempt:
        attempt = Attempt(status="", record=None)
        src, tgt = direction.split("->")
        try:
            gen = self._ask(
                self.generator,
                self.prompts.render("generate_single", lang_name=LANG_NAMES[src], passage=chunk[src]),
                attempt,
            )
            if is_true(gen.get("skip")):
                raise _Rejected("generator_skipped")
            q, a, evidence = (str(gen.get(k, "")).strip() for k in ("question", "answer", "evidence"))
            if not (q and a and evidence):
                raise _Rejected("missing_fields")
            if not contains_span(chunk[src], evidence, src):
                raise _Rejected("evidence_not_in_passage")
            if not contains_span(chunk[src], a, src):
                raise _Rejected("answer_not_in_passage")
            if has_context_reference(q, src):
                raise _Rejected("context_reference")

            q_tgt, a_tgt = self._translate(src, tgt, q, a, attempt)

            span = self._verify(tgt, q_tgt, [chunk[tgt]], attempt)
            if span is None:
                raise _Rejected("not_answerable_translated")
            if not contains_span(chunk[tgt], span, tgt):
                raise _Rejected("verified_answer_not_in_passage")
            if token_f1(span, a_tgt, tgt) < self.f1_threshold:
                raise _Rejected("verified_answer_mismatch")
        except _Rejected as rejected:
            attempt.status = rejected.reason
            return attempt

        attempt.status = "accepted"
        attempt.record = {
            "source": "unpc",
            "type": "single",
            "direction": direction,
            "translation": "machine",
            "question": {src: q, tgt: q_tgt},
            "answer": {src: a, tgt: span},
            "answer_translation": {tgt: a_tgt},
            "gold_chunks": [chunk["chunk_id"]],
            "evidence": {chunk["chunk_id"]: {src: evidence}},
            "llm_calls": attempt.calls,
        }
        return attempt

    def bridge(self, chunk_a: dict, chunk_b: dict, symbol: str, direction: str) -> Attempt:
        attempt = Attempt(status="", record=None)
        src, tgt = direction.split("->")
        try:
            gen = self._ask(
                self.generator,
                self.prompts.render(
                    "generate_bridge",
                    lang_name=LANG_NAMES[src],
                    symbol=symbol,
                    passage_a=chunk_a[src],
                    passage_b=chunk_b[src],
                ),
                attempt,
            )
            if is_true(gen.get("skip")):
                raise _Rejected("generator_skipped")
            q, a, ev_a, ev_b = (str(gen.get(k, "")).strip() for k in ("question", "answer", "evidence_a", "evidence_b"))
            if not (q and a and ev_a and ev_b):
                raise _Rejected("missing_fields")
            if not contains_span(chunk_a[src], ev_a, src) or not contains_span(chunk_b[src], ev_b, src):
                raise _Rejected("evidence_not_in_passage")
            if not contains_span(chunk_b[src], a, src):
                raise _Rejected("answer_not_in_passage_b")
            if has_context_reference(q, src):
                raise _Rejected("context_reference")
            if mentions_document_number(q, src):
                raise _Rejected("document_number_in_question")

            q_tgt, a_tgt = self._translate(src, tgt, q, a, attempt)
            if mentions_document_number(q_tgt, tgt):
                raise _Rejected("document_number_in_question_translated")

            q_en, a_en = (q, a) if src == "en" else (q_tgt, a_tgt)
            for name, passage in (("b", chunk_b["en"]), ("a", chunk_a["en"])):
                span = self._verify("en", q_en, [passage], attempt)
                if span is not None and token_f1(span, a_en, "en") >= self.f1_threshold:
                    raise _Rejected(f"shortcut_{name}_alone")

            span = self._verify(tgt, q_tgt, [chunk_a[tgt], chunk_b[tgt]], attempt)
            if span is None:
                raise _Rejected("not_answerable_translated")
            if not contains_span(chunk_b[tgt], span, tgt):
                raise _Rejected("verified_answer_not_in_passage_b")
            if token_f1(span, a_tgt, tgt) < self.f1_threshold:
                raise _Rejected("verified_answer_mismatch")
        except _Rejected as rejected:
            attempt.status = rejected.reason
            return attempt

        attempt.status = "accepted"
        attempt.record = {
            "source": "unpc",
            "type": "bridge",
            "direction": direction,
            "translation": "machine",
            "bridge_symbol": symbol,
            "question": {src: q, tgt: q_tgt},
            "answer": {src: a, tgt: span},
            "answer_translation": {tgt: a_tgt},
            "gold_chunks": [chunk_a["chunk_id"], chunk_b["chunk_id"]],
            "evidence": {chunk_a["chunk_id"]: {src: ev_a}, chunk_b["chunk_id"]: {src: ev_b}},
            "llm_calls": attempt.calls,
        }
        return attempt


def replace_kinds(previous: list[dict], new: list[dict], run_kinds: set[str], field: str) -> list[dict]:
    """Keep earlier entries of kinds not run this time; entries of kinds that were run are replaced."""
    return [entry for entry in previous if entry[field] not in run_kinds] + new


@dataclass
class BuildResult:
    accepted: list[dict]
    outcomes: list[dict]
    usage: Counter
    stopped: str | None  # None, or why the run stopped early


def run_build(kind: str, attempt_fn, candidates: Iterable, n_target: int, max_candidates: int) -> BuildResult:
    """Try candidates in order until `n_target` are accepted, balancing translation direction."""
    quota = {d: (n_target + i) // 2 for i, d in enumerate(DIRECTIONS)}  # en->ar gets the extra one if odd
    accepted_by_dir: Counter[str] = Counter()
    result = BuildResult([], [], Counter(), None)
    for i, candidate in enumerate(candidates):
        if len(result.accepted) >= n_target:
            break
        if i >= max_candidates:
            result.stopped = f"max_candidates ({max_candidates}) reached"
            break
        direction = DIRECTIONS[i % 2]
        if accepted_by_dir[direction] >= quota[direction]:
            direction = DIRECTIONS[(i + 1) % 2]
        try:
            attempt = attempt_fn(candidate, direction)
        except DailyLimitReached as limit:
            result.stopped = f"daily limit: {limit}"
            break
        result.usage.update(attempt.usage)
        chunk_ids = [c["chunk_id"] for c in (candidate if isinstance(candidate, tuple) else (candidate,)) if isinstance(c, dict)]
        result.outcomes.append(
            {"kind": kind, "candidate": i, "chunks": chunk_ids, "direction": direction, "status": attempt.status, "llm_calls": attempt.calls}
        )
        if attempt.record is not None:
            accepted_by_dir[direction] += 1
            result.accepted.append(attempt.record)
    else:
        if len(result.accepted) < n_target:
            result.stopped = "candidates exhausted"
    return result
