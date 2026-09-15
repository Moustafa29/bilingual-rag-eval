"""Read single members of a large remote zip archive with HTTP range requests.

A zip file ends with a central directory: a table of every member's name, byte offset and
compressed size. Reading that table once (15-20 MB for the OPUS UN corpus archives) is enough to
fetch any single document with two small range requests, instead of downloading a 1.6-1.9 GB
archive.
"""

from __future__ import annotations

import io
import json
import struct
import time
import zipfile
import zlib
from pathlib import Path

import httpx

_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_HEADER_SIGNATURE = 0x04034B50


class _RangeReader(io.RawIOBase):
    """Seekable file object over a remote file; lets `zipfile` read the central directory."""

    def __init__(self, remote: RemoteZip):
        self.remote = remote
        self.size = remote.size()
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.remote.get_range(self.pos, self.pos + n - 1)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


class RemoteZip:
    def __init__(
        self,
        url: str,
        index_path: str | Path,
        client: httpx.Client | None = None,
        retries: int = 4,
    ):
        self.url = url
        self.client = client or httpx.Client(follow_redirects=True, timeout=120)
        self.retries = retries
        index_path = Path(index_path)
        if index_path.exists():
            self.index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            self.index = self._read_central_directory()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(self.index), encoding="utf-8")

    def size(self) -> int:
        response = self._request("HEAD", {})
        return int(response.headers["content-length"])

    def get_range(self, start: int, end: int) -> bytes:
        response = self._request("GET", {"Range": f"bytes={start}-{end}"})
        if response.status_code != 206:
            raise OSError(f"range request not honoured ({response.status_code}) for {self.url}")
        expected = end - start + 1
        if len(response.content) != expected:
            raise OSError(f"short read: got {len(response.content)} of {expected} bytes")
        return response.content

    def _request(self, method: str, headers: dict) -> httpx.Response:
        for attempt in range(self.retries):
            try:
                response = self.client.request(method, self.url, headers=headers)
            except httpx.TransportError:
                if attempt == self.retries - 1:
                    raise
            else:
                if response.status_code < 500:
                    response.raise_for_status()
                    return response
                if attempt == self.retries - 1:
                    response.raise_for_status()
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _read_central_directory(self) -> dict[str, list[int]]:
        reader = io.BufferedReader(_RangeReader(self), buffer_size=1 << 16)
        with zipfile.ZipFile(reader) as archive:
            return {
                info.filename: [
                    info.header_offset,
                    info.compress_size,
                    info.file_size,
                    info.compress_type,
                    info.CRC,
                ]
                for info in archive.infolist()
                if not info.is_dir()
            }

    def read(self, name: str) -> bytes:
        offset, compressed_size, size, method, crc = self.index[name]
        header = _LOCAL_HEADER.unpack(self.get_range(offset, offset + _LOCAL_HEADER.size - 1))
        if header[0] != _LOCAL_HEADER_SIGNATURE:
            raise OSError(f"bad local header for {name}")
        name_length, extra_length = header[-2], header[-1]
        start = offset + _LOCAL_HEADER.size + name_length + extra_length
        raw = self.get_range(start, start + compressed_size - 1) if compressed_size else b""
        if method == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(raw, -15)
        elif method == zipfile.ZIP_STORED:
            data = raw
        else:
            raise OSError(f"unsupported compression method {method} for {name}")
        if len(data) != size or zlib.crc32(data) != crc:
            raise OSError(f"CRC or size mismatch for {name}")
        return data
