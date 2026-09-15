"""Run retrieval configurations over a corpus and write ranked lists (top 100) per query.

    python scripts/run_retrieval.py --corpus xquad --configs bm25-raw,bm25-norm,bm25-light
    python scripts/run_retrieval.py --corpus xquad --configs e5-base,bge-m3 --cross-lingual
    python scripts/run_retrieval.py --corpus xquad --configs hybrid:bm25-light+e5-base,rerank:hybrid:bm25-light+e5-base

Config names:
  bm25-<raw|norm|light>          lexical, same language as the query
  <e5-base|bge-m3>               dense
  hybrid:<a>+<b>                 RRF (k=60) over the top 100 of two existing runs
  rerank:<run>                   cross-encoder over the top 50 of an existing run

Runs are written to data/runs/<corpus>/<config>/<query_lang>-<doc_lang>.jsonl. Dense runs with
--cross-lingual also write en-ar and ar-en (Arabic queries over English passages and the reverse).
Hybrid and rerank configs read the runs they combine, so run those first.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rageval.eval.corpora import QUESTION_FILES
from rageval.io import read_jsonl, write_jsonl
from rageval.retrieval.analyzers import analyze
from rageval.retrieval.bm25 import BM25
from rageval.retrieval.dense import MODELS, Encoder, exact_search
from rageval.retrieval.fusion import reciprocal_rank_fusion

DEPTH = 100
RERANK_DEPTH = 50
LANGS = ("en", "ar")


def corpus_paths(data: Path, corpus: str) -> tuple[Path, Path]:
    return data / "corpus" / corpus / "chunks.jsonl", data / "questions" / QUESTION_FILES[corpus]


def run_path(data: Path, corpus: str, config: str, qlang: str, dlang: str) -> Path:
    return data / "runs" / corpus / config.replace(":", "_").replace("+", "__") / f"{qlang}-{dlang}.jsonl"


def write_run(path: Path, qids: list[str], rankings: list[list[tuple[str, float]]], seconds: float) -> None:
    write_jsonl(path, [{"qid": q, "ranking": [[c, round(s, 6)] for c, s in r]} for q, r in zip(qids, rankings)])
    print(f"  wrote {path} ({seconds:.1f}s, {seconds / max(len(qids), 1) * 1000:.1f} ms/query)")


def read_run(path: Path) -> dict[str, list[str]]:
    return {r["qid"]: [c for c, _ in r["ranking"]] for r in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=sorted(QUESTION_FILES), required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--cross-lingual", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = Path(args.data)
    chunks_path, questions_path = corpus_paths(data, args.corpus)
    chunks = read_jsonl(chunks_path)
    questions = read_jsonl(questions_path)
    chunk_ids = [c["chunk_id"] for c in chunks]
    text_by_id = {c["chunk_id"]: c for c in chunks}
    qids = [q["qid"] for q in questions]

    for config in args.configs.split(","):
        print(config)
        if config.startswith("bm25-"):
            analyzer = config.removeprefix("bm25-")
            for lang in LANGS:
                start = time.perf_counter()
                index = BM25([analyze(c[lang], lang, analyzer) for c in chunks])
                rankings = [
                    [(chunk_ids[i], s) for i, s in index.search(analyze(q["question"][lang], lang, analyzer), DEPTH)]
                    for q in questions
                ]
                write_run(run_path(data, args.corpus, config, lang, lang), qids, rankings, time.perf_counter() - start)

        elif config in MODELS:
            encoder = Encoder(MODELS[config], data / "embeddings" / args.corpus, batch_size=args.batch_size, device=args.device)
            passages = {lang: encoder.encode([c[lang] for c in chunks], "passage") for lang in LANGS}
            pairs = [(l, l) for l in LANGS] + ([("ar", "en"), ("en", "ar")] if args.cross_lingual else [])
            for qlang, dlang in pairs:
                start = time.perf_counter()
                query_vectors = encoder.encode([q["question"][qlang] for q in questions], "query")
                ids, scores = exact_search(passages[dlang], query_vectors, DEPTH)
                rankings = [[(chunk_ids[i], float(s)) for i, s in zip(row_ids, row_scores) if i >= 0] for row_ids, row_scores in zip(ids, scores)]
                write_run(run_path(data, args.corpus, config, qlang, dlang), qids, rankings, time.perf_counter() - start)

        elif config.startswith("hybrid:"):
            parts = config.removeprefix("hybrid:").split("+")
            for lang in LANGS:
                start = time.perf_counter()
                runs = [read_run(run_path(data, args.corpus, part, lang, lang)) for part in parts]
                rankings = [reciprocal_rank_fusion([run[q] for run in runs], k=60, depth=DEPTH)[:DEPTH] for q in qids]
                write_run(run_path(data, args.corpus, config, lang, lang), qids, rankings, time.perf_counter() - start)

        elif config.startswith("rerank:"):
            from rageval.retrieval.rerank import Reranker

            base = config.removeprefix("rerank:")
            reranker = Reranker(device=args.device)
            for lang in LANGS:
                start = time.perf_counter()
                run = read_run(run_path(data, args.corpus, base, lang, lang))
                rankings = []
                for q in questions:
                    shortlist = run[q["qid"]][:RERANK_DEPTH]
                    rankings.append(reranker.rerank(q["question"][lang], [(cid, text_by_id[cid][lang]) for cid in shortlist]))
                write_run(run_path(data, args.corpus, config, lang, lang), qids, rankings, time.perf_counter() - start)
        else:
            raise SystemExit(f"unknown config: {config}")


if __name__ == "__main__":
    main()
