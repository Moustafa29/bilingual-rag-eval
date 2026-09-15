"""LLM question generation with verification. Every rejection is recorded with its reason.

Single-hop, per candidate chunk (also the numeric group, with a numeric-answer prompt):
  1. generate  (generator, source language, source passage): question, short answer, evidence
     sentence. Rule checks: evidence occurs in the passage (tolerant of clitic and hyphen copy
     variants), the answer occurs verbatim, no "this document" phrasing.
  2. translate (generator, *no passage*): question and answer into the target language. The
     passage is withheld so the translation cannot borrow its wording, which would inflate
     lexical overlap and favour BM25 in the target language.
  3. verify    (verifier, a different model, target language, target passage): answer the
     translated question from the passage alone. The span must occur in the passage.
  4. judge     (verifier): does the verified target span state the same fact as the source answer?
     This replaced a token-F1 >= 0.5 gate between the verified span and the blind translation. That
     gate accepted a wrong answer in the pilot: "the Habitat Agenda" came back as "the UN Human
     Settlements Programme" and passed on the shared words "human settlements" (F1 0.75). F1 is
     still recorded, but no longer decides.

Comparison (two-hop), per candidate (two documents indexed under the same subject term):
  1. generate from both source passages: a question naming two subjects, one per passage, whose
     answer is one of the two.
  2. translate question and options.
  3. shortcut checks in English: the verifier must NOT pick an option from either passage alone.
     Any pick from one passage rejects the candidate, whether or not it matches.
  4. verify from both target passages by choosing an option; the choice must match. Choosing
     among options needs no fuzzy answer matching. Options are shown in a content-hashed order
     so a verifier's position bias cannot systematically pass or fail candidates.

Bridge (two-hop via document citation) is kept for the documented negative result: prompts show a
header naming each passage's document symbol, and the shortcut rule is strict as above.

Direction alternates en->ar / ar->en so that neither language always receives the machine-
translated question. Once one direction's quota is full, the other is used.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from rageval.corpus.numbers import has_grouped_number
from rageval.llm.client import DailyLimitReached
from rageval.questions.checks import has_context_reference, is_true, mentions_document_number, parse_json_object
from rageval.text import contains_evidence, contains_span, normalize, token_f1

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


def with_header(chunk: dict, lang: str) -> str:
    return f"{chunk['header']}\n{chunk[lang]}"


@dataclass
class Attempt:
    status: str  # "accepted" or the rejection reason
    record: dict | None
    calls: list[str] = field(default_factory=list)
    usage: Counter = field(default_factory=Counter)
    question: dict = field(default_factory=dict)  # generated question text, kept for rejections too (audit)


class _Rejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class QuestionBuilder:
    def __init__(self, generator, verifier, prompts: Prompts):
        self.generator = generator
        self.verifier = verifier
        self.prompts = prompts

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

    def _same_answer(self, question: str, src: str, answer_src: str, tgt: str, answer_tgt: str, attempt: Attempt) -> bool:
        prompt = self.prompts.render(
            "judge_equivalence",
            src_lang_name=LANG_NAMES[src],
            tgt_lang_name=LANG_NAMES[tgt],
            question=question,
            answer_src=answer_src,
            answer_tgt=answer_tgt,
        )
        return is_true(self._ask(self.verifier, prompt, attempt).get("same"))

    def _choose(self, lang: str, question: str, options: list[str], passages: list[str], attempt: Attempt) -> int | None:
        """1 or 2 as numbered in `options`, or None if the verifier cannot decide from the passages."""
        swap = int(hashlib.sha256("\x00".join([question, *passages]).encode("utf-8")).hexdigest(), 16) % 2 == 1
        shown = options[::-1] if swap else options
        prompt = self.prompts.render(
            "verify_choice",
            lang_name=LANG_NAMES[lang],
            question=question,
            option_1=shown[0],
            option_2=shown[1],
            passages=format_passages(passages),
        )
        choice = self._ask(self.verifier, prompt, attempt).get("choice")
        if str(choice) not in ("1", "2"):
            return None
        return 3 - int(choice) if swap else int(choice)

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
        attempt.question[tgt] = q
        if has_context_reference(q, tgt):
            raise _Rejected("context_reference_translated")
        return q, a

    def single(
        self, chunk: dict, direction: str, kind: str = "single", prompt: str = "generate_single", require_number: bool = False
    ) -> Attempt:
        attempt = Attempt(status="", record=None)
        src, tgt = direction.split("->")
        try:
            gen = self._ask(self.generator, self.prompts.render(prompt, lang_name=LANG_NAMES[src], passage=chunk[src]), attempt)
            if is_true(gen.get("skip")):
                raise _Rejected("generator_skipped")
            q, a, evidence = (str(gen.get(k, "")).strip() for k in ("question", "answer", "evidence"))
            if not (q and a and evidence):
                raise _Rejected("missing_fields")
            attempt.question[src] = q
            if not contains_evidence(chunk[src], evidence, src):
                raise _Rejected("evidence_not_in_passage")
            if not contains_span(chunk[src], a, src):
                raise _Rejected("answer_not_in_passage")
            if has_context_reference(q, src):
                raise _Rejected("context_reference")
            if require_number and src == "en" and not has_grouped_number(a):
                raise _Rejected("answer_not_a_thousands_number")

            q_tgt, a_tgt = self._translate(src, tgt, q, a, attempt)
            if require_number and src == "ar" and not has_grouped_number(a_tgt):
                raise _Rejected("answer_not_a_thousands_number")

            span = self._verify(tgt, q_tgt, [chunk[tgt]], attempt)
            if span is None:
                raise _Rejected("not_answerable_translated")
            if not contains_span(chunk[tgt], span, tgt):
                raise _Rejected("verified_answer_not_in_passage")
            f1 = token_f1(span, a_tgt, tgt)
            if not self._same_answer(q, src, a, tgt, span, attempt):
                raise _Rejected("answer_not_equivalent")
        except _Rejected as rejected:
            attempt.status = rejected.reason
            return attempt

        attempt.status = "accepted"
        attempt.record = {
            "source": "unpc",
            "kind": kind,
            "type": "single",
            "direction": direction,
            "translation": "machine",
            "question": {src: q, tgt: q_tgt},
            "answer": {src: a, tgt: span},
            "answer_translation": {tgt: a_tgt},
            "answer_f1_vs_translation": round(f1, 3),
            "gold_chunks": [chunk["chunk_id"]],
            "evidence": {chunk["chunk_id"]: {src: evidence}},
            "llm_calls": attempt.calls,
        }
        return attempt

    def comparison(self, chunk_1: dict, chunk_2: dict, keyword: str, direction: str) -> Attempt:
        attempt = Attempt(status="", record=None)
        src, tgt = direction.split("->")
        try:
            gen = self._ask(
                self.generator,
                self.prompts.render("generate_comparison", lang_name=LANG_NAMES[src], passage_1=chunk_1[src], passage_2=chunk_2[src]),
                attempt,
            )
            if is_true(gen.get("skip")):
                raise _Rejected("generator_skipped")
            q = str(gen.get("question", "")).strip()
            options = gen.get("options")
            ev_1, ev_2 = str(gen.get("evidence_1", "")).strip(), str(gen.get("evidence_2", "")).strip()
            if not q or not isinstance(options, list) or len(options) != 2 or str(gen.get("answer")) not in ("1", "2"):
                raise _Rejected("missing_fields")
            options = [str(o).strip() for o in options]
            index = int(gen["answer"])
            if not all(options) or not (ev_1 and ev_2):
                raise _Rejected("missing_fields")
            attempt.question[src] = q
            if normalize(options[0], src) == normalize(options[1], src):
                raise _Rejected("options_not_distinct")
            if not contains_evidence(chunk_1[src], ev_1, src) or not contains_evidence(chunk_2[src], ev_2, src):
                raise _Rejected("evidence_not_in_passage")
            if not all(contains_evidence(q, option, src) for option in options):
                raise _Rejected("options_not_in_question")
            if has_context_reference(q, src):
                raise _Rejected("context_reference")

            out = self._ask(
                self.generator,
                self.prompts.render(
                    "translate_comparison",
                    src_lang_name=LANG_NAMES[src],
                    tgt_lang_name=LANG_NAMES[tgt],
                    question=q,
                    option_1=options[0],
                    option_2=options[1],
                ),
                attempt,
            )
            q_tgt = str(out.get("question", "")).strip()
            options_tgt = out.get("options")
            if not q_tgt or not isinstance(options_tgt, list) or len(options_tgt) != 2:
                raise _Rejected("translation_missing_fields")
            options_tgt = [str(o).strip() for o in options_tgt]
            if not all(options_tgt):
                raise _Rejected("translation_missing_fields")
            attempt.question[tgt] = q_tgt
            if has_context_reference(q_tgt, tgt):
                raise _Rejected("context_reference_translated")

            q_en, options_en = (q, options) if src == "en" else (q_tgt, options_tgt)
            for name, chunk in (("1", chunk_1), ("2", chunk_2)):
                if self._choose("en", q_en, options_en, [chunk["en"]], attempt) is not None:
                    raise _Rejected(f"shortcut_{name}_alone")

            choice = self._choose(tgt, q_tgt, options_tgt, [chunk_1[tgt], chunk_2[tgt]], attempt)
            if choice is None:
                raise _Rejected("not_answerable_translated")
            if choice != index:
                raise _Rejected("verified_choice_mismatch")
        except _Rejected as rejected:
            attempt.status = rejected.reason
            return attempt

        attempt.status = "accepted"
        attempt.record = {
            "source": "unpc",
            "kind": "comparison",
            "type": "comparison",
            "direction": direction,
            "translation": "machine",
            "shared_keyword": keyword,
            "question": {src: q, tgt: q_tgt},
            "options": {src: options, tgt: options_tgt},
            "answer_index": index,
            "answer": {src: options[index - 1], tgt: options_tgt[index - 1]},
            "gold_chunks": [chunk_1["chunk_id"], chunk_2["chunk_id"]],
            "evidence": {chunk_1["chunk_id"]: {src: ev_1}, chunk_2["chunk_id"]: {src: ev_2}},
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
                    header_a=chunk_a["header"],
                    header_b=chunk_b["header"],
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
            attempt.question[src] = q
            if not contains_evidence(chunk_a[src], ev_a, src) or not contains_evidence(chunk_b[src], ev_b, src):
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

            q_en = q if src == "en" else q_tgt
            for name, chunk in (("b", chunk_b), ("a", chunk_a)):
                if self._verify("en", q_en, [with_header(chunk, "en")], attempt) is not None:
                    raise _Rejected(f"shortcut_{name}_alone")

            span = self._verify(tgt, q_tgt, [with_header(chunk_a, tgt), with_header(chunk_b, tgt)], attempt)
            if span is None:
                raise _Rejected("not_answerable_translated")
            if not contains_span(chunk_b[tgt], span, tgt):
                raise _Rejected("verified_answer_not_in_passage_b")
            f1 = token_f1(span, a_tgt, tgt)
            if not self._same_answer(q, src, a, tgt, span, attempt):
                raise _Rejected("answer_not_equivalent")
        except _Rejected as rejected:
            attempt.status = rejected.reason
            return attempt

        attempt.status = "accepted"
        attempt.record = {
            "source": "unpc",
            "kind": "bridge",
            "type": "bridge",
            "direction": direction,
            "translation": "machine",
            "bridge_symbol": symbol,
            "question": {src: q, tgt: q_tgt},
            "answer": {src: a, tgt: span},
            "answer_translation": {tgt: a_tgt},
            "answer_f1_vs_translation": round(f1, 3),
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
            {
                "kind": kind,
                "candidate": i,
                "chunks": chunk_ids,
                "direction": direction,
                "status": attempt.status,
                "question": attempt.question,
                "llm_calls": attempt.calls,
            }
        )
        if attempt.record is not None:
            accepted_by_dir[direction] += 1
            result.accepted.append(attempt.record)
    else:
        if len(result.accepted) < n_target:
            result.stopped = "candidates exhausted"
    return result
