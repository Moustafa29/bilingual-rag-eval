"""Download XQuAD (en, ar) and write it in the same passage/question format as the UN corpus.

    python scripts/build_xquad.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from rageval.corpus.chunking import TokenCounter
from rageval.corpus.xquad import load_xquad
from rageval.io import load_config, write_jsonl

URL = "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.{lang}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = Path(cfg["paths"]["data"])
    raw = data / "xquad"
    for lang in ("en", "ar"):
        path = raw / f"xquad.{lang}.json"
        if not path.exists():
            raw.mkdir(parents=True, exist_ok=True)
            response = httpx.get(URL.format(lang=lang), follow_redirects=True, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)

    passages, questions = load_xquad(raw / "xquad.en.json", raw / "xquad.ar.json")
    ch = cfg["chunking"]
    counter = TokenCounter(ch["tokenizer"])
    for lang in ("en", "ar"):
        for p, n in zip(passages, counter([p[lang] for p in passages], prefix=ch["passage_prefix"], special_tokens=True)):
            p[f"{lang}_tokens"] = n
    for p in passages:
        p["truncated_e5"] = max(p["en_tokens"], p["ar_tokens"]) > ch["model_max_tokens"]

    write_jsonl(data / "corpus" / "xquad" / "chunks.jsonl", passages)
    write_jsonl(data / "questions" / "xquad.jsonl", questions)
    stats = {
        "passages": len(passages),
        "questions": len(questions),
        "truncated_e5": sum(p["truncated_e5"] for p in passages),
        "max_tokens": {lang: max(p[f"{lang}_tokens"] for p in passages) for lang in ("en", "ar")},
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
