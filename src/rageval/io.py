from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal KEY=VALUE loader so API keys can live in an untracked .env file."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl_field(path: str | Path, field: str) -> list:
    """One field from every record, without holding the file in memory.

    The UN chunk file is ~104 MB and holds both languages; loading all of it to use a few hundred
    passages exhausted a 7.7 GB laptop during generation.
    """
    values = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                values.append(json.loads(line)[field])
    return values


def read_jsonl_subset(path: str | Path, keys: Iterable[str], key_field: str, value_field: str) -> dict:
    """{key_field: value_field} for the records whose key is wanted; the rest are parsed and dropped."""
    wanted = set(keys)
    found = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record[key_field] in wanted:
                found[record[key_field]] = record[value_field]
                if len(found) == len(wanted):
                    break
    return found


def jsonl_fingerprint(path: str | Path) -> str:
    """SHA-256 over a file's lines with line endings removed.

    The same records written on Windows (CRLF) and on Linux/Colab (LF) get the same fingerprint, so a
    corpus rebuilt on Colab can be checked against one built on the laptop.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for line in f:
            digest.update(line.rstrip(b"\r\n"))
            digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)
