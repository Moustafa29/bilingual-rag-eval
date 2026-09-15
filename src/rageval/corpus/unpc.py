"""UN Parallel Corpus subsample: pair index, deterministic selection, fetching.

Selection rule, in order:

1. Eligible pairs. Documents present in both English and Arabic whose alignment is mostly
   one-to-one, within a size range (thresholds in the config). Poorly aligned pairs would give
   chunks whose two language versions do not say the same thing.
2. Seed documents. Eligible documents sorted by `sha256("<salt>|<doc_id>")`, first N taken.
   Hash order is a random order anyone can recompute without a seed-dependent RNG, and adding
   or removing an unrelated document does not reshuffle the rest.
3. Cited documents. Eligible documents referenced by symbol in a seed's English text, in the
   same hash order, up to a cap. These make bridge questions possible: a random sample of a
   few thousand out of ~110,000 documents rarely contains both ends of a reference.

The resulting document ids are committed in `manifests/unpc_docs.tsv`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rageval.corpus.remote_zip import RemoteZip
from rageval.corpus.tei import Link, doc_id_from_path, iter_link_groups

LANGS = ("en", "ar")


@dataclass(frozen=True)
class PairStats:
    doc_id: str
    doc_score: float | None
    n_links: int
    n_one_to_one: int

    @property
    def one_to_one_ratio(self) -> float:
        return self.n_one_to_one / self.n_links if self.n_links else 0.0


def build_pair_index(alignment_gz: Path, out_tsv: Path) -> dict[str, int]:
    counts = {"pairs": 0, "path_mismatch": 0}
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_tsv.with_suffix(".tmp")
    with gzip.open(alignment_gz, "rt", encoding="utf-8") as lines, open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["doc_id", "doc_score", "n_links", "n_one_to_one"])
        for group in iter_link_groups(lines):
            ar_id = doc_id_from_path(group.ar_path, "ar")
            en_id = doc_id_from_path(group.en_path, "en")
            if ar_id != en_id:
                counts["path_mismatch"] += 1
                continue
            n_11 = sum(link.one_to_one for link in group.links)
            writer.writerow([en_id, "" if group.score is None else group.score, len(group.links), n_11])
            counts["pairs"] += 1
    tmp.replace(out_tsv)
    return counts


def read_pair_index(path: Path) -> list[PairStats]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            PairStats(
                doc_id=row["doc_id"],
                doc_score=float(row["doc_score"]) if row["doc_score"] else None,
                n_links=int(row["n_links"]),
                n_one_to_one=int(row["n_one_to_one"]),
            )
            for row in csv.DictReader(f, delimiter="\t")
        ]


def is_eligible(pair: PairStats, rules: dict) -> bool:
    return (
        rules["min_one_to_one_links"] <= pair.n_one_to_one <= rules["max_one_to_one_links"]
        and pair.one_to_one_ratio >= rules["min_one_to_one_ratio"]
    )


def selection_key(salt: str, doc_id: str) -> str:
    return hashlib.sha256(f"{salt}|{doc_id}".encode("utf-8")).hexdigest()


def select_seeds(doc_ids: Iterable[str], salt: str, n: int) -> list[str]:
    return sorted(doc_ids, key=lambda d: selection_key(salt, d))[:n]


def cited_closure(
    seeds: list[str],
    cited_keys: dict[str, set[str]],
    key_to_docs: dict[str, list[str]],
    salt: str,
    max_added: int,
) -> list[str]:
    seed_set = set(seeds)
    candidates = {
        doc
        for seed in seeds
        for key in cited_keys.get(seed, ())
        for doc in key_to_docs.get(key, ())
        if doc not in seed_set
    }
    return sorted(candidates, key=lambda d: selection_key(salt, d))[:max_added]


def extract_links(alignment_gz: Path, doc_ids: set[str]) -> dict[str, list[Link]]:
    def keep(ar_path: str, en_path: str) -> bool:
        doc_id = doc_id_from_path(en_path, "en")
        return doc_id in doc_ids and doc_id_from_path(ar_path, "ar") == doc_id

    found = {}
    with gzip.open(alignment_gz, "rt", encoding="utf-8") as lines:
        for group in iter_link_groups(lines, keep=keep):
            found[doc_id_from_path(group.en_path, "en")] = group.links
    return found


def write_links(path: Path, links: dict[str, list[Link]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc_id in sorted(links):
            record = {"doc_id": doc_id, "links": [[list(l.ar), list(l.en)] for l in links[doc_id]]}
            f.write(json.dumps(record) + "\n")


def read_links(path: Path) -> dict[str, list[Link]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            out[record["doc_id"]] = [Link(ar=tuple(ar), en=tuple(en)) for ar, en in record["links"]]
    return out


def raw_path(raw_dir: Path, lang: str, doc_id: str) -> Path:
    return raw_dir / lang / f"{doc_id}.xml"


def fetch_documents(
    remote: RemoteZip, lang: str, doc_ids: list[str], raw_dir: Path, workers: int = 8
) -> dict[str, str]:
    """Fetch documents not yet on disk. Returns doc_id -> "cached" | "fetched" | "missing"."""

    def fetch(doc_id: str) -> tuple[str, str]:
        path = raw_path(raw_dir, lang, doc_id)
        if path.exists():
            return doc_id, "cached"
        member = f"UNPC/raw/{lang}/{doc_id}.xml"
        if member not in remote.index:
            return doc_id, "missing"
        data = remote.read(member)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return doc_id, "fetched"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(fetch, doc_ids))
