"""OpenAI-compatible chat client with a disk cache and a usage ledger.

Every response is stored on disk, keyed by the SHA-256 of the endpoint, model, messages and
generation parameters. Repeating a call with the same inputs is served from disk at no cost, so
question-set builds and evaluations can be re-run for free; only a changed prompt or parameter
reaches the network. Network calls are appended to a JSONL ledger, which is what cost estimates
are checked against.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx


class LLMError(RuntimeError):
    pass


class DailyLimitReached(LLMError):
    """The provider's daily quota is spent. Cached work is kept; re-run after the reset."""


@dataclass
class ChatResult:
    text: str
    cache_key: str
    cached: bool
    usage: dict


def cache_key(base_url: str, model: str, messages: list[dict], params: dict) -> str:
    payload = json.dumps(
        {"base_url": base_url, "model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, key: str, record: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        params: dict,
        cache: DiskCache,
        ledger_path: str | Path,
        api_key: str | None,
        http: httpx.Client | None = None,
        max_retries: int = 8,
        max_wait_s: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.params = params
        self.cache = cache
        self.ledger_path = Path(ledger_path)
        self.api_key = api_key
        self.http = http or httpx.Client(timeout=180)
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.sleep = sleep

    @classmethod
    def from_config(cls, role: dict, cache: DiskCache, ledger_path: str | Path) -> ChatClient:
        return cls(
            base_url=role["base_url"],
            model=role["model"],
            params=role.get("params", {}),
            cache=cache,
            ledger_path=ledger_path,
            api_key=os.environ.get(role["api_key_env"]),
        )

    def chat(self, messages: list[dict]) -> ChatResult:
        key = cache_key(self.base_url, self.model, messages, self.params)
        hit = self.cache.get(key)
        if hit is not None:
            return ChatResult(hit["response"]["text"], key, True, hit["response"].get("usage", {}))
        if not self.api_key:
            raise LLMError(f"no API key for {self.model} and the response is not cached")

        body = {"model": self.model, "messages": messages, **self.params}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries):
            response = self.http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                result = {
                    "text": choice["message"].get("content") or "",
                    "usage": data.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                }
                break
            if response.status_code == 400 and "json_validate_failed" in response.text:
                # The model produced invalid JSON in JSON mode. Cache the failure so re-runs are
                # deterministic and free; the caller treats empty text as unparseable.
                result = {"text": "", "usage": {}, "error": response.text[:1000]}
                break
            if response.status_code == 429:
                message = response.text.lower()
                if "per day" in message or "(tpd)" in message or "(rpd)" in message:
                    raise DailyLimitReached(response.text[:500])
                wait = float(response.headers.get("retry-after", 2**attempt))
                if wait > self.max_wait_s:
                    raise DailyLimitReached(f"retry-after {wait}s: {response.text[:300]}")
                self.sleep(wait)
                continue
            if response.status_code >= 500:
                self.sleep(2**attempt)
                continue
            raise LLMError(f"{self.model}: HTTP {response.status_code}: {response.text[:500]}")
        else:
            raise LLMError(f"{self.model}: retries exhausted")

        self.cache.put(
            key,
            {
                "request": {"base_url": self.base_url, "model": self.model, "messages": messages, "params": self.params},
                "response": result,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        self._record_usage(key, result["usage"])
        return ChatResult(result["text"], key, False, result["usage"])

    def _record_usage(self, key: str, usage: dict) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "base_url": self.base_url,
            "model": self.model,
            "key": key,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
