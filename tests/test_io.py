from rageval.io import jsonl_fingerprint


def test_fingerprint_ignores_line_endings_but_not_content(tmp_path):
    lf, crlf, other = tmp_path / "lf.jsonl", tmp_path / "crlf.jsonl", tmp_path / "other.jsonl"
    lf.write_bytes(b'{"a": 1}\n{"b": "\xd9\x86\xd8\xb5"}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"b": "\xd9\x86\xd8\xb5"}\r\n')
    other.write_bytes(b'{"a": 2}\n{"b": "\xd9\x86\xd8\xb5"}\n')
    assert jsonl_fingerprint(lf) == jsonl_fingerprint(crlf)
    assert jsonl_fingerprint(lf) != jsonl_fingerprint(other)


def test_read_jsonl_field_and_subset(tmp_path):
    from rageval.io import read_jsonl_field, read_jsonl_subset

    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '{"chunk_id": "a", "en": "one", "ar": "واحد"}\n'
        '{"chunk_id": "b", "en": "two", "ar": "اثنان"}\n'
        '{"chunk_id": "c", "en": "three", "ar": "ثلاثة"}\n',
        encoding="utf-8",
    )
    assert read_jsonl_field(path, "chunk_id") == ["a", "b", "c"]
    assert read_jsonl_subset(path, {"c", "a"}, "chunk_id", "en") == {"a": "one", "c": "three"}
    assert read_jsonl_subset(path, {"b"}, "chunk_id", "ar") == {"b": "اثنان"}
    assert read_jsonl_subset(path, set(), "chunk_id", "en") == {}
