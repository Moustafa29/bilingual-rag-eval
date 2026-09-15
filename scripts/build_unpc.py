"""Build the UN Parallel Corpus subsample and its aligned chunks. Every step is resumable.

Steps: pair index -> eligibility -> seed selection -> fetch -> citation closure -> fetch ->
manifest -> links -> chunks -> stats.

    python scripts/build_unpc.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

from rageval.corpus.chunking import AlignmentError, TokenCounter, build_beads, make_chunks
from rageval.corpus.remote_zip import RemoteZip
from rageval.corpus.symbols import doc_key, find_references, symbol_key
from rageval.corpus.tei import parse_tei
from rageval.corpus.unpc import (
    LANGS,
    build_pair_index,
    cited_closure,
    extract_links,
    fetch_documents,
    is_eligible,
    raw_path,
    read_links,
    read_pair_index,
    select_seeds,
    write_links,
)
from rageval.io import load_config, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response, open(tmp, "wb") as f:
        response.raise_for_status()
        for block in response.iter_bytes(1 << 20):
            f.write(block)
    tmp.replace(path)


def fetch_all(remotes: dict[str, RemoteZip], doc_ids: list[str], raw_dir: Path) -> set[str]:
    """Fetch both languages; return the ids available in both."""
    available = set(doc_ids)
    for lang in LANGS:
        status = fetch_documents(remotes[lang], lang, doc_ids, raw_dir)
        counts = Counter(status.values())
        print(f"  {lang}: {dict(counts)}")
        available &= {d for d, s in status.items() if s != "missing"}
    return available


def percentiles(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values)
    return {p: float(np.percentile(arr, int(p[1:]))) for p in ("p50", "p90", "p99")} | {"max": int(arr.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    u, ch = cfg["unpc"], cfg["chunking"]
    data = Path(cfg["paths"]["data"])
    unpc_dir, raw_dir = data / "unpc", data / "unpc" / "raw"
    stats: dict = {}

    alignment = Path(u["alignment_file"])
    if not alignment.exists():
        print(f"downloading alignment index -> {alignment}")
        download(f"{u['opus_base']}/xml/ar-en.xml.gz", alignment)
    digest = sha256_file(alignment)
    if u.get("alignment_sha256") and digest != u["alignment_sha256"]:
        raise SystemExit(f"alignment file checksum {digest} does not match config")
    stats["alignment_sha256"] = digest

    index_path = unpc_dir / "pairs.tsv"
    if not index_path.exists():
        print("building pair index (streams the whole alignment file once)")
        print(" ", build_pair_index(alignment, index_path))
    pairs = read_pair_index(index_path)
    eligible = [p for p in pairs if is_eligible(p, u["eligibility"])]
    eligible_ids = {p.doc_id for p in eligible}
    stats["pairs"] = len(pairs)
    stats["eligible_pairs"] = len(eligible)
    print(f"pairs {len(pairs)}, eligible {len(eligible)}")

    remotes = {lang: RemoteZip(f"{u['opus_base']}/raw/{lang}.zip", unpc_dir / f"zip_index_{lang}.json") for lang in LANGS}

    salt = u["selection_salt"]
    seeds = select_seeds(eligible_ids, salt, u["n_seed_docs"])
    print(f"fetching {len(seeds)} seed documents")
    available = fetch_all(remotes, seeds, raw_dir)
    seeds = [d for d in seeds if d in available]

    key_to_docs: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(eligible_ids):
        key_to_docs[doc_key(doc_id)].append(doc_id)
    cited: dict[str, set[str]] = {}
    for doc_id in seeds:
        doc = parse_tei(raw_path(raw_dir, "en", doc_id).read_bytes())
        cited[doc_id] = {symbol_key(ref) for ref in find_references("\n".join(s.text for s in doc.sentences))}
    added = cited_closure(seeds, cited, key_to_docs, salt, u["max_cited_docs"])
    print(f"fetching {len(added)} cited documents")
    available = fetch_all(remotes, added, raw_dir)
    added = [d for d in added if d in available]

    selected = [(d, "seed") for d in seeds] + [(d, "cited") for d in added]
    manifest = Path(cfg["paths"]["manifests"]) / "unpc_docs.tsv"
    rows = [[d, role] for d, role in selected]
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            committed = [row for row in csv.reader(f, delimiter="\t")][1:]
        if committed != rows:
            raise SystemExit(f"selection differs from committed {manifest}; delete it only if the rule changed on purpose")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["doc_id", "role"])
            writer.writerows(rows)
    doc_ids = [d for d, _ in selected]
    stats["selected_docs"] = {"seed": len(seeds), "cited": len(added)}

    links_path = unpc_dir / "links.jsonl"
    links = read_links(links_path) if links_path.exists() else {}
    if set(doc_ids) - set(links):
        print("extracting sentence links (streams the alignment file once)")
        links = extract_links(alignment, set(doc_ids))
        write_links(links_path, links)

    counter = TokenCounter(ch["tokenizer"])
    dropped: Counter[str] = Counter()
    symbol_rule = Counter()
    chunks: list[dict] = []
    for doc_id in tqdm(doc_ids, desc="chunking"):
        en = parse_tei(raw_path(raw_dir, "en", doc_id).read_bytes())
        ar = parse_tei(raw_path(raw_dir, "ar", doc_id).read_bytes())
        if en.symbol:
            symbol_rule["match" if symbol_key(en.symbol) == doc_key(doc_id) else "mismatch"] += 1
        if doc_id not in links:
            dropped["no_links"] += 1
            continue
        try:
            beads = build_beads(en, ar, links[doc_id])
        except AlignmentError:
            dropped["alignment_error"] += 1
            continue
        doc_chunks = [
            c for c in make_chunks(doc_id, beads, counter, ch["target_tokens"], ch["max_tokens"]) if c["en"].strip() and c["ar"].strip()
        ]
        if not doc_chunks:
            dropped["no_chunks"] += 1
            continue
        chunks.extend(doc_chunks)

    prefix, limit = ch["passage_prefix"], ch["model_max_tokens"]
    for lang in LANGS:
        counts = counter([c[lang] for c in chunks], prefix=prefix, special_tokens=True)
        for c, n in zip(chunks, counts):
            c[f"{lang}_tokens"] = n
    for c in chunks:
        c["truncated_e5"] = max(c["en_tokens"], c["ar_tokens"]) > limit

    out_dir = data / "corpus" / "unpc"
    write_jsonl(out_dir / "chunks.jsonl", chunks)
    ratios = [c["ar_tokens"] / c["en_tokens"] for c in chunks]
    stats.update(
        {
            "dropped_docs": dict(dropped),
            "symbol_rule": dict(symbol_rule),
            "chunks": len(chunks),
            "docs_with_chunks": len({c["doc_id"] for c in chunks}),
            "tokens_en": percentiles([c["en_tokens"] for c in chunks]),
            "tokens_ar": percentiles([c["ar_tokens"] for c in chunks]),
            "ar_to_en_token_ratio": {"median": float(np.median(ratios)), "mean": float(np.mean(ratios))},
            "truncated_e5": sum(c["truncated_e5"] for c in chunks),
        }
    )
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
