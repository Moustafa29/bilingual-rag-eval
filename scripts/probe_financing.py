"""Probe: can comparison questions be built deterministically from UN mission-financing resolutions?

No LLM calls. The gate below was fixed and committed before the probe was first run (0f18ff4).

A comparison question needs two missions whose budgets are comparable. General Assembly resolutions
titled "Financing of the <mission>" appropriate an amount for a stated budget period.

Definitions
- Appropriation record: a paragraph of a financing resolution that appropriates (not apportions) an
  amount for a period, extracted by `rageval.corpus.financing`, whose amount also appears in an Arabic
  paragraph about an amount (مبلغ) in the aligned Arabic resolution, after the digit-group correction.
  Additional appropriations and reductions are excluded. One record per (canonical mission, period).
- Usable pair: two records for different missions with the identical budget period. Each record is
  used in at most one pair (n missions in a period give floor(n/2) pairs, not n*(n-1)/2).

Gate: at least 60 usable pairs -> build comparison questions. Fewer -> no multi-hop set from this
corpus. The loose count (all same-period combinations) is reported but does not decide.

Measurement history (the gate never changed):
- Run 1 (0f18ff4): 113 usable pairs, but 3 of 10 sampled records were wrong (amounts written in words,
  an apportionment read as an appropriation, truncated mission titles).
- Run 2 (7c046c1): 108 usable pairs; 5 of 30 sampled records were wrong or not comparable (two
  support-account shares, one of them named only "Mission"; three partial-year appropriations).
- Run 3 (this version): support-account shares and partial-year appropriations are excluded, and a
  mission name taken from a paragraph needs 3+ words.

Acceptance of the count, fixed before run 3: a fresh seeded sample of 30 records (--sample-seed, not
the seed used for run 2) must contain at most 1 wrong or non-comparable record. Otherwise the count is
not trusted, whatever it is.

    python scripts/probe_financing.py --config configs/pilot.yaml --sample-seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rageval.corpus.financing import arabic_confirms, canonical_mission, document_mission, extract_appropriation
from rageval.corpus.numbers import correct_reversed_numbers
from rageval.corpus.remote_zip import RemoteZip
from rageval.corpus.tei import parse_tei
from rageval.corpus.unpc import LANGS, fetch_documents, raw_path, read_pair_index
from rageval.io import load_config, write_jsonl

GATE = 60
FINANCING_TITLE = re.compile(r"\bFinancing of the [A-Z]")


def paragraphs(doc) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for s in doc.sentences:
        if s.pid not in grouped:
            order.append(s.pid)
        grouped[s.pid].append(s.text)
    return [" ".join(grouped[p]) for p in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--sample-seed", type=int, default=20260915)
    args = parser.parse_args()
    cfg = load_config(args.config)
    u = cfg["unpc"]
    data = Path(cfg["paths"]["data"])
    unpc_dir, raw_dir = data / "unpc", data / "unpc" / "raw"
    remotes = {lang: RemoteZip(f"{u['opus_base']}/raw/{lang}.zip", unpc_dir / f"zip_index_{lang}.json") for lang in LANGS}

    resolutions = sorted(p.doc_id for p in read_pair_index(unpc_dir / "pairs.tsv") if re.match(r"^\d{4}/a/res/", p.doc_id))
    status = fetch_documents(remotes["en"], "en", resolutions, raw_dir)
    print(f"GA resolutions with an English-Arabic pair: {len(resolutions)}; en: {dict(Counter(status.values()))}")

    financing = {}
    for doc_id in resolutions:
        path = raw_path(raw_dir, "en", doc_id)
        if path.exists():
            doc = parse_tei(path.read_bytes())
            if any(FINANCING_TITLE.search(s.text) for s in doc.sentences[:40]):
                financing[doc_id] = doc
    status = fetch_documents(remotes["ar"], "ar", sorted(financing), raw_dir)
    print(f"financing resolutions: {len(financing)}; ar: {dict(Counter(status.values()))}")

    counts: Counter[str] = Counter()
    records: dict[tuple[str, str], dict] = {}
    for doc_id, en_doc in sorted(financing.items()):
        ar_path = raw_path(raw_dir, "ar", doc_id)
        if not ar_path.exists():
            counts["no_arabic_document"] += 1
            continue
        en_paragraphs = paragraphs(en_doc)
        en_text = "\n".join(en_paragraphs)
        doc_mission = document_mission(" ".join(en_paragraphs))
        ar_paragraphs = correct_reversed_numbers(en_text, "\n".join(paragraphs(parse_tei(ar_path.read_bytes()))))[0].split("\n")
        for paragraph in en_paragraphs:
            record = extract_appropriation(paragraph)
            if record is None:
                continue
            counts["english_appropriation_paragraphs"] += 1
            if record["kind"] != "appropriation":
                counts[f"excluded_{record['kind']}"] += 1
                continue
            mission = record["mission_in_paragraph"] or doc_mission
            if not mission:
                counts["no_mission_name"] += 1
                continue
            if not arabic_confirms(ar_paragraphs, record["amount_text"]):
                counts["arabic_amount_not_found"] += 1
                continue
            key = (canonical_mission(mission), record["period"])
            if key in records:
                counts["duplicate_mission_period"] += 1
                continue
            records[key] = {
                "mission": mission,
                "mission_key": key[0],
                "period": record["period"],
                "amount_usd": record["amount_usd"],
                "amount_text": record["amount_text"],
                "doc_id": doc_id,
                "english": paragraph,
                "arabic": next((p for p in ar_paragraphs if arabic_confirms([p], record["amount_text"])), ""),
            }

    by_period: dict[str, list[dict]] = defaultdict(list)
    for record in records.values():
        by_period[record["period"]].append(record)
    usable = sum(len(v) // 2 for v in by_period.values())
    loose = sum(len(v) * (len(v) - 1) // 2 for v in by_period.values())

    out = data / "probe"
    ordered = sorted(records.values(), key=lambda r: (r["period"], r["mission_key"]))
    write_jsonl(out / "financing_records.jsonl", ordered)
    sample = random.Random(args.sample_seed).sample(ordered, min(30, len(ordered)))
    lines = [f"# Seeded sample of 30 appropriation records, seed {args.sample_seed} (manual check)", ""]
    for r in sample:
        lines += [f"## {r['mission']} | {r['period']} | {r['amount_text']} = {r['amount_usd']:,} | {r['doc_id']}", f"- EN: {r['english'][:450]}", f"- AR: {r['arabic'][:450]}", ""]
    (out / f"sample_records_seed{args.sample_seed}.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "ga_resolutions_scanned": len(resolutions),
        "financing_resolutions": len(financing),
        **dict(counts),
        "records_one_per_mission_period": len(records),
        "distinct_missions": len({r["mission_key"] for r in records.values()}),
        "distinct_periods": len(by_period),
        "missions_per_period": dict(sorted(Counter(len(v) for v in by_period.values()).items())),
        "usable_pairs_disjoint_same_period": usable,
        "loose_pairs_all_same_period_combinations": loose,
        "gate": GATE,
        "verdict": "build comparison questions" if usable >= GATE else "below gate: no multi-hop set from this corpus",
    }
    (out / "financing_probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
