"""Probe: can comparison questions be built deterministically from UN mission-financing resolutions?

No LLM calls. Run before building anything; the gate below was fixed before the probe was run.

A comparison question needs two missions whose budgets are comparable. General Assembly resolutions
titled "Financing of the <mission>" appropriate an amount for a stated budget period ("Decides to
appropriate ... the amount of 760,567,400 dollars ... for the period from 1 July 2005 to 30 June 2006").

Definitions
- Appropriation record: an English paragraph of a financing resolution stating an appropriation,
  an amount and a period "from <date> to <date>", whose amount also appears as a grouped number in
  an Arabic paragraph mentioning an amount (مبلغ) in the aligned Arabic resolution, after the Arabic
  digit-group correction. Records are kept one per (mission, period).
- Usable pair: two records for different missions with the identical budget period. Each record is
  used in at most one pair, so the count cannot be inflated by combinations (n missions in a period
  give floor(n/2) pairs, not n*(n-1)/2).

Gate: at least 60 usable pairs -> build comparison questions from them. Fewer -> no multi-hop set
from this corpus. The loose count (all same-period combinations) is reported alongside for
transparency, but does not decide.

    python scripts/probe_financing.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rageval.corpus.numbers import correct_reversed_numbers
from rageval.corpus.remote_zip import RemoteZip
from rageval.corpus.tei import parse_tei
from rageval.corpus.unpc import LANGS, fetch_documents, raw_path, read_pair_index
from rageval.io import load_config, write_jsonl
from rageval.text import normalize

GATE = 60

TITLE = re.compile(r"\bFinancing of the ([A-Z][^\n;:]{5,120}?)(?:\s*\(continued\))?\s*$")
APPROPRIATION = re.compile(r"\bappropriat", re.IGNORECASE)
AMOUNT = re.compile(r"\bamount of (?:US)?\$?\s?(\d{1,3}(?:,\d{3})+)")
PERIOD = re.compile(r"\bperiod from (\d{1,2} [A-Z][a-z]+ \d{4}) to (\d{1,2} [A-Z][a-z]+ \d{4})")
AR_AMOUNT_WORD = normalize("مبلغ", "ar")


def paragraphs(doc) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for s in doc.sentences:
        if s.pid not in grouped:
            order.append(s.pid)
        grouped[s.pid].append(s.text)
    return [" ".join(grouped[p]) for p in order]


def arabic_has_amount(ar_paragraphs: list[str], groups: list[str]) -> bool:
    forms = [" ".join(groups), ",".join(groups), "".join(groups)]
    for paragraph in ar_paragraphs:
        text = normalize(paragraph, "ar")
        if AR_AMOUNT_WORD in text and any(re.search(rf"(?<!\d){re.escape(f)}(?!\d)", text) for f in forms):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    u = cfg["unpc"]
    data = Path(cfg["paths"]["data"])
    unpc_dir, raw_dir = data / "unpc", data / "unpc" / "raw"
    remotes = {lang: RemoteZip(f"{u['opus_base']}/raw/{lang}.zip", unpc_dir / f"zip_index_{lang}.json") for lang in LANGS}

    resolutions = sorted(p.doc_id for p in read_pair_index(unpc_dir / "pairs.tsv") if re.match(r"^\d{4}/a/res/", p.doc_id))
    print(f"GA resolutions with an English-Arabic pair: {len(resolutions)}; fetching English")
    status = fetch_documents(remotes["en"], "en", resolutions, raw_dir)
    print("  en:", dict(Counter(status.values())))

    financing = {}
    for doc_id in resolutions:
        path = raw_path(raw_dir, "en", doc_id)
        if not path.exists():
            continue
        doc = parse_tei(path.read_bytes())
        head = [s.text for s in doc.sentences[:40]]
        title = next((m.group(1).strip() for t in head if (m := TITLE.search(t))), None)
        if title:
            financing[doc_id] = (title, doc)
    print(f"financing resolutions: {len(financing)}; fetching Arabic")
    status = fetch_documents(remotes["ar"], "ar", sorted(financing), raw_dir)
    print("  ar:", dict(Counter(status.values())))

    counts = Counter()
    records: dict[tuple[str, str], dict] = {}
    for doc_id, (title, en_doc) in sorted(financing.items()):
        ar_path = raw_path(raw_dir, "ar", doc_id)
        if not ar_path.exists():
            counts["no_arabic_document"] += 1
            continue
        en_paragraphs = paragraphs(en_doc)
        en_text = "\n".join(en_paragraphs)
        ar_paragraphs = paragraphs(parse_tei(ar_path.read_bytes()))
        corrected = correct_reversed_numbers(en_text, "\n".join(ar_paragraphs))[0].split("\n")
        found_in_doc = False
        for paragraph in en_paragraphs:
            if not APPROPRIATION.search(paragraph):
                continue
            amount, period = AMOUNT.search(paragraph), PERIOD.search(paragraph)
            if not (amount and period):
                continue
            counts["english_appropriation_paragraphs"] += 1
            groups = amount.group(1).split(",")
            if not arabic_has_amount(corrected, groups):
                counts["arabic_amount_not_found"] += 1
                continue
            key = (title.lower(), f"{period.group(1)} - {period.group(2)}")
            if key not in records:
                records[key] = {
                    "mission": title,
                    "period": key[1],
                    "amount_usd": int(amount.group(1).replace(",", "")),
                    "doc_id": doc_id,
                    "english": paragraph,
                }
            found_in_doc = True
        counts["documents_with_a_record" if found_in_doc else "documents_without_a_record"] += 1

    by_period: dict[str, list[dict]] = defaultdict(list)
    for record in records.values():
        by_period[record["period"]].append(record)
    usable = sum(len(v) // 2 for v in by_period.values())
    loose = sum(len(v) * (len(v) - 1) // 2 for v in by_period.values())

    out = data / "probe"
    write_jsonl(out / "financing_records.jsonl", sorted(records.values(), key=lambda r: (r["period"], r["mission"])))
    summary = {
        "ga_resolutions_scanned": len(resolutions),
        "financing_resolutions": len(financing),
        **dict(counts),
        "records_one_per_mission_period": len(records),
        "distinct_missions": len({r["mission"].lower() for r in records.values()}),
        "distinct_periods": len(by_period),
        "missions_per_period": dict(Counter(len(v) for v in by_period.values())),
        "usable_pairs_disjoint_same_period": usable,
        "loose_pairs_all_same_period_combinations": loose,
        "gate": GATE,
        "verdict": "build comparison questions" if usable >= GATE else "below gate: no multi-hop set from this corpus",
    }
    (out / "financing_probe.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
