import io
import zipfile

import httpx
import pytest

from rageval.corpus.remote_zip import RemoteZip

MEMBERS = {"UNPC/raw/en/1992/a/1.xml": "hello " * 500, "UNPC/raw/ar/1992/a/1.xml": "مرحبا"}


def make_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("UNPC/raw/en/1992/a/1.xml", MEMBERS["UNPC/raw/en/1992/a/1.xml"], zipfile.ZIP_DEFLATED)
        archive.writestr("UNPC/raw/ar/1992/a/1.xml", MEMBERS["UNPC/raw/ar/1992/a/1.xml"], zipfile.ZIP_STORED)
        archive.writestr("UNPC/raw/en/", "")
    return buffer.getvalue()


def range_server(blob: bytes, log: list):
    def handler(request: httpx.Request) -> httpx.Response:
        log.append((request.method, request.headers.get("range")))
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(blob))})
        start, end = (int(x) for x in request.headers["range"].removeprefix("bytes=").split("-"))
        return httpx.Response(206, content=blob[start : end + 1])

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_reads_members_through_range_requests(tmp_path):
    blob, log = make_zip(), []
    remote = RemoteZip("https://example.org/en.zip", tmp_path / "index.json", client=range_server(blob, log))
    assert set(remote.index) == set(MEMBERS)  # directories are skipped
    for name, text in MEMBERS.items():
        assert remote.read(name).decode("utf-8") == text


def test_index_is_cached_on_disk(tmp_path):
    blob, log = make_zip(), []
    RemoteZip("https://example.org/en.zip", tmp_path / "index.json", client=range_server(blob, log))
    log.clear()
    remote = RemoteZip("https://example.org/en.zip", tmp_path / "index.json", client=range_server(blob, log))
    assert log == []
    remote.read("UNPC/raw/ar/1992/a/1.xml")
    assert len(log) == 2  # local header + member data


def test_corrupt_member_is_detected(tmp_path):
    blob = make_zip()
    remote = RemoteZip("https://example.org/en.zip", tmp_path / "index.json", client=range_server(blob, []))
    offset, csize, size, method, crc = remote.index["UNPC/raw/ar/1992/a/1.xml"]
    remote.index["UNPC/raw/ar/1992/a/1.xml"] = [offset, csize, size, method, crc ^ 1]
    with pytest.raises(OSError):
        remote.read("UNPC/raw/ar/1992/a/1.xml")
