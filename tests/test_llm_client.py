import json

import httpx
import pytest

from rageval.llm.client import ChatClient, DailyLimitReached, DiskCache, LLMError

OK = {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def make_client(tmp_path, responses, sleeps=None, params=None, api_key="key"):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return responses[min(len(calls), len(responses)) - 1]

    client = ChatClient(
        base_url="https://api.example.org/v1",
        model="m",
        params=params or {"temperature": 0},
        cache=DiskCache(tmp_path / "cache"),
        ledger_path=tmp_path / "ledger.jsonl",
        api_key=api_key,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )
    return client, calls


MESSAGES = [{"role": "user", "content": "hi"}]


def test_second_identical_call_is_served_from_cache(tmp_path):
    client, calls = make_client(tmp_path, [httpx.Response(200, json=OK)])
    first = client.chat(MESSAGES)
    second = client.chat(MESSAGES)
    assert (first.cached, second.cached) == (False, True)
    assert first.text == second.text == '{"a": 1}'
    assert len(calls) == 1
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1


def test_changed_params_miss_the_cache(tmp_path):
    client, calls = make_client(tmp_path, [httpx.Response(200, json=OK)])
    client.chat(MESSAGES)
    client.params = {"temperature": 0.5}
    client.chat(MESSAGES)
    assert len(calls) == 2


def test_cached_response_needs_no_api_key(tmp_path):
    client, _ = make_client(tmp_path, [httpx.Response(200, json=OK)])
    client.chat(MESSAGES)
    offline, calls = make_client(tmp_path, [httpx.Response(500)], api_key=None)
    assert offline.chat(MESSAGES).cached
    assert calls == []
    offline.params = {"temperature": 1}
    with pytest.raises(LLMError):
        offline.chat(MESSAGES)


def test_rate_limit_waits_for_retry_after(tmp_path):
    sleeps = []
    responses = [httpx.Response(429, headers={"retry-after": "3"}, text="Rate limit reached on tokens per minute (TPM)"), httpx.Response(200, json=OK)]
    client, calls = make_client(tmp_path, responses, sleeps=sleeps)
    assert client.chat(MESSAGES).text == '{"a": 1}'
    assert sleeps == [3.0] and len(calls) == 2


def test_daily_limit_raises(tmp_path):
    client, _ = make_client(tmp_path, [httpx.Response(429, text="Rate limit reached on tokens per day (TPD)")])
    with pytest.raises(DailyLimitReached):
        client.chat(MESSAGES)


def test_invalid_json_generation_is_cached_as_empty(tmp_path):
    client, calls = make_client(tmp_path, [httpx.Response(400, text='{"error": {"code": "json_validate_failed"}}')])
    assert client.chat(MESSAGES).text == ""
    assert client.chat(MESSAGES).cached and len(calls) == 1


def test_other_client_errors_raise(tmp_path):
    client, _ = make_client(tmp_path, [httpx.Response(401, text="bad key")])
    with pytest.raises(LLMError):
        client.chat(MESSAGES)
