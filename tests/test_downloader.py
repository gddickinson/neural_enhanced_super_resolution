"""Regression tests for the model downloader: correct upstream URL, size and SHA-256 for the Real-ESRGAN weights,
and hash verification (no network: requests.get is replaced).

The table used to list RealESRGAN_x2plus.pth under release v0.2.5.0 (404: the file is in v0.2.1) with MD5 values that
matched neither upstream file, so a download could never verify. Upstream values checked 2026-11-12 by downloading
github.com/xinntao/Real-ESRGAN/releases v0.2.1/RealESRGAN_x2plus.pth and v0.1.0/RealESRGAN_x4plus.pth.
"""
import hashlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nesr.utils import downloader  # noqa: E402

WEIGHTS = os.path.join(ROOT, "models", "weights", "RealESRGAN_x2plus.pth")


def test_x2plus_entry_matches_upstream_and_committed_file():
    m = downloader.MODELS["esrgan_x2"]
    assert m["url"] == "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    assert m["size"] == 67061725
    assert m["sha256"] == "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
    if os.path.exists(WEIGHTS):  # the committed copy is the upstream file
        assert os.path.getsize(WEIGHTS) == m["size"]
        assert downloader.calculate_sha256(WEIGHTS) == m["sha256"]


def test_x4plus_entry():
    m = downloader.MODELS["esrgan_x4"]
    assert m["url"].endswith("/v0.1.0/RealESRGAN_x4plus.pth")
    assert m["size"] == 67040989
    assert m["sha256"] == "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


class _Resp:
    def __init__(self, body):
        self.body, self.headers, self.status_code = body, {"content-length": str(len(body))}, 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


@pytest.fixture
def served(monkeypatch):
    calls = []

    def fake_get(url, headers=None, stream=True, timeout=None):
        calls.append(dict(headers or {}))
        return _Resp(served.body)
    served.body = b"weights" * 1000
    monkeypatch.setattr(downloader.requests, "get", fake_get)
    served.calls = calls
    return served


def test_download_verifies_sha256(tmp_path, served):
    dest = tmp_path / "w.pth"
    good = hashlib.sha256(served.body).hexdigest()
    assert downloader.download_file("u", str(dest), expected_size=len(served.body), expected_sha256=good)
    assert dest.read_bytes() == served.body
    dest.unlink()
    assert not downloader.download_file("u", str(dest), expected_size=len(served.body), expected_sha256="0" * 64)


def test_corrupt_complete_file_is_downloaded_again(tmp_path, served):
    dest = tmp_path / "w.pth"
    dest.write_bytes(b"x" * len(served.body))  # right size, wrong content
    good = hashlib.sha256(served.body).hexdigest()
    assert downloader.download_file("u", str(dest), expected_size=len(served.body), expected_sha256=good)
    assert "Range" not in served.calls[-1]  # fresh download, not a resume past the end
    assert dest.read_bytes() == served.body


def test_verified_existing_file_is_not_downloaded(tmp_path, served):
    dest = tmp_path / "w.pth"
    dest.write_bytes(served.body)
    good = hashlib.sha256(served.body).hexdigest()
    assert downloader.download_file("u", str(dest), expected_size=len(served.body), expected_sha256=good)
    assert served.calls == []
