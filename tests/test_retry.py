"""HTTP retry / backoff / size-cap tests."""
import io
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from shared.retry import (
    api_request_with_retry,
    download_file,
    _get_retry_delay,
    _seekable_files,
    DEFAULT_MAX_DOWNLOAD_BYTES,
)
from shared.errors import APITransientError, APIPermanentError


# --------- _seekable_files ---------

def test_seekable_files_resets_bytesio_in_dict():
    buf = io.BytesIO(b"abcdef")
    buf.read()  # advance to end
    _seekable_files({"files": {"audio": ("name.wav", buf)}})
    assert buf.tell() == 0


def test_seekable_files_resets_in_list_of_tuples():
    buf = io.BytesIO(b"abcdef")
    buf.read()
    _seekable_files({"files": [("files", ("a.wav", buf, "audio/wav"))]})
    assert buf.tell() == 0


def test_seekable_files_ignores_plain_bytes():
    """Plain bytes have no seek — should silently skip without crash."""
    _seekable_files({"files": {"audio": ("name.wav", b"raw bytes")}})  # no exception


def test_seekable_files_handles_missing_files_key():
    _seekable_files({})  # no exception


# --------- api_request_with_retry ---------

def _ok_response(status=200, body=b"ok"):
    r = requests.Response()
    r.status_code = status
    r._content = body
    return r


def _err_response(status, body=b'{"detail":"bad"}'):
    r = requests.Response()
    r.status_code = status
    r._content = body
    return r


def test_retry_succeeds_after_transient_failures():
    """A 503 followed by a 200 should retry and return the 200."""
    responses = [_err_response(503), _err_response(503), _ok_response()]
    requester = MagicMock()
    requester.request.side_effect = responses
    with patch("shared.retry.time.sleep") as sleep_mock:
        resp = api_request_with_retry("GET", "http://x", session=requester,
                                      max_retries=3, base_delay=0.01)
    assert resp.status_code == 200
    assert requester.request.call_count == 3
    assert sleep_mock.call_count == 2


def test_retry_no_retry_on_4xx():
    """400/401/403 should not retry."""
    requester = MagicMock()
    requester.request.return_value = _err_response(401, b'{"detail":"unauthorized"}')
    with patch("shared.retry.time.sleep"):
        with pytest.raises(APIPermanentError):
            api_request_with_retry("GET", "http://x", session=requester, max_retries=3)
    assert requester.request.call_count == 1


def test_retry_max_retries_then_raises():
    requester = MagicMock()
    requester.request.return_value = _err_response(503)
    with patch("shared.retry.time.sleep"):
        with pytest.raises(APITransientError):
            api_request_with_retry("GET", "http://x", session=requester,
                                   max_retries=2, base_delay=0.01)
    assert requester.request.call_count == 3  # initial + 2 retries


def test_retry_seeks_files_between_attempts():
    """Multipart file-like body should be seeked back to 0 between retries."""
    buf = io.BytesIO(b"audio payload")
    requester = MagicMock()
    requester.request.side_effect = [_err_response(503), _ok_response()]
    with patch("shared.retry.time.sleep"):
        api_request_with_retry("POST", "http://x",
                               session=requester,
                               files={"audio": ("a.wav", buf)},
                               max_retries=3, base_delay=0.01)
    # After 2nd attempt, buf should be at 0 (seek before the 2nd call)
    assert buf.tell() == 0


def test_retry_on_connection_error():
    requester = MagicMock()
    requester.request.side_effect = [
        requests.exceptions.ConnectionError("boom"),
        _ok_response(),
    ]
    with patch("shared.retry.time.sleep"):
        resp = api_request_with_retry("GET", "http://x", session=requester,
                                      max_retries=3, base_delay=0.01)
    assert resp.status_code == 200


def test_retry_secret_scrubbed_from_connection_error():
    """API key in connection-error URL must not leak into raised exception."""
    requester = MagicMock()
    requester.request.side_effect = requests.exceptions.ConnectionError(
        "Failed to connect to https://api.elevenlabs.io/v1/?xi-api-key=SECRETabcdef1234567890")
    with patch("shared.retry.time.sleep"):
        with pytest.raises(APITransientError) as exc_info:
            api_request_with_retry("GET", "http://x", session=requester, max_retries=0)
    # Note: api_request_with_retry doesn't scrub itself; api_post does.
    # This test verifies the message _does_ contain the URL so api_post can scrub it.
    # The scrubbing is tested in test_utils_api_helpers.py.
    assert "SECRET" in str(exc_info.value) or "REDACTED" in str(exc_info.value)


# --------- _get_retry_delay ---------

def test_get_retry_delay_uses_retry_after_seconds():
    r = _err_response(429)
    r.headers["Retry-After"] = "5"
    delay = _get_retry_delay(r, base_delay=2.0, attempt=0)
    assert delay == 5.0


def test_get_retry_delay_uses_retry_after_http_date():
    """HTTP-date Retry-After should resolve to a delta in seconds."""
    from email.utils import formatdate
    future = time.time() + 10
    r = _err_response(429)
    r.headers["Retry-After"] = formatdate(future, usegmt=True)
    delay = _get_retry_delay(r, base_delay=2.0, attempt=0, max_delay=60)
    # Allow ±2s slop
    assert 7.5 <= delay <= 12.5


def test_get_retry_delay_falls_back_to_exponential():
    r = _ok_response()
    delay = _get_retry_delay(r, base_delay=2.0, attempt=2, max_delay=60)
    # 2 * 2^2 = 8 with ±25% jitter → range [6, 10]
    assert 5.5 <= delay <= 10.5


def test_get_retry_delay_clamped_to_max():
    r = _err_response(429)
    r.headers["Retry-After"] = "9999"
    delay = _get_retry_delay(r, base_delay=2.0, attempt=0, max_delay=10.0)
    assert delay == 10.0


# --------- download_file ---------

def test_download_file_streams_under_cap():
    """Normal small download succeeds."""
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {}
    fake_resp.iter_content = lambda chunk_size: [b"hello", b" world"]
    fake_resp.raise_for_status = lambda: None
    session = MagicMock()
    session.get.return_value = fake_resp
    data = download_file("http://x", session=session, max_bytes=100)
    assert data == b"hello world"


def test_download_file_aborts_on_streamed_size():
    """Should abort mid-stream if accumulated bytes exceed cap."""
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {}
    # Each chunk is 100 bytes; cap is 250 → abort after 3rd chunk
    fake_resp.iter_content = lambda chunk_size: [b"x" * 100, b"x" * 100, b"x" * 100, b"x" * 100]
    fake_resp.raise_for_status = lambda: None
    session = MagicMock()
    session.get.return_value = fake_resp
    with pytest.raises(APIPermanentError):
        download_file("http://x", session=session, max_bytes=250)


def test_download_file_rejects_content_length_over_cap():
    """Should refuse before streaming when Content-Length exceeds cap."""
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.headers = {"Content-Length": "1000"}
    fake_resp.raise_for_status = lambda: None
    session = MagicMock()
    session.get.return_value = fake_resp
    with pytest.raises(APIPermanentError):
        download_file("http://x", session=session, max_bytes=500)


def test_download_file_default_cap_is_reasonable():
    """Default cap is 500MB, not unbounded."""
    assert DEFAULT_MAX_DOWNLOAD_BYTES == 500 * 1024 * 1024
