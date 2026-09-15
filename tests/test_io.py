from rageval.io import jsonl_fingerprint


def test_fingerprint_ignores_line_endings_but_not_content(tmp_path):
    lf, crlf, other = tmp_path / "lf.jsonl", tmp_path / "crlf.jsonl", tmp_path / "other.jsonl"
    lf.write_bytes(b'{"a": 1}\n{"b": "\xd9\x86\xd8\xb5"}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"b": "\xd9\x86\xd8\xb5"}\r\n')
    other.write_bytes(b'{"a": 2}\n{"b": "\xd9\x86\xd8\xb5"}\n')
    assert jsonl_fingerprint(lf) == jsonl_fingerprint(crlf)
    assert jsonl_fingerprint(lf) != jsonl_fingerprint(other)
