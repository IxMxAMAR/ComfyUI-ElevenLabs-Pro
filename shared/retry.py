"""HTTP request helpers with exponential-backoff retry logic."""

import io
import random
import time
from email.utils import parsedate_to_datetime
from typing import Optional

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout as ReqTimeout
from requests.exceptions import ChunkedEncodingError as ReqChunkedEncodingError
from requests.exceptions import RequestException as ReqRequestException

from .errors import APITransientError, APIPermanentError, parse_error_response


# Hard cap on download size to prevent DoS / disk-fill from a malicious /
# misconfigured response. 500 MB is well above any legitimate ElevenLabs
# output (10-min WAV @ 48k stereo = ~110 MB).
DEFAULT_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024


def _seekable_files(kwargs):
    """Reset all file-like objects in kwargs['files'] to position 0.

    `requests` reads the file object during send; on retry it would re-send
    empty body unless we seek back. Tuple-of-bytes form is fine (bytes are
    re-readable) but file objects need reset. Best-effort: silently skip
    items that don't support seek.
    """
    files = kwargs.get("files")
    if not files:
        return
    if isinstance(files, dict):
        entries = list(files.values())
    else:
        # list of (field_name, value) tuples — extract the values
        entries = []
        for item in files:
            if isinstance(item, tuple) and len(item) == 2:
                # (field_name, value) — value can be file-like or a (filename, file, ...) tuple
                entries.append(item[1])
            else:
                entries.append(item)

    for entry in entries:
        # entry shapes per requests docs:
        #   bytes / str / file-like
        #   (filename, file_or_bytes)
        #   (filename, file_or_bytes, content_type)
        #   (filename, file_or_bytes, content_type, headers)
        candidate = entry
        if isinstance(entry, tuple) and len(entry) >= 2:
            candidate = entry[1]
        try:
            seek = getattr(candidate, "seek", None)
            if callable(seek):
                seek(0)
        except Exception:
            pass


def api_request_with_retry(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    timeout: float = 60,
    transient_codes: tuple = (429, 500, 502, 503, 504),
    service_name: str = "API",
    **kwargs,
) -> requests.Response:
    """Make an HTTP request with exponential backoff retry on transient failures.

    Retries on 429 and 5xx (configurable via `transient_codes`) and on
    connection / timeout / chunked-encoding errors. Honors `Retry-After`
    header in both seconds and HTTP-date format. Adds jitter to backoff
    to avoid thundering herd.

    File-like objects in `files=` are seeked back to 0 before each retry
    so multipart uploads survive transient failures.
    """
    requester = session or requests
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        # Reset file pointers before each attempt (not just retries —
        # cheap and bulletproof).
        if attempt > 0:
            _seekable_files(kwargs)

        try:
            response = requester.request(method, url, timeout=timeout, **kwargs)

            if response.status_code < 400:
                return response

            body = response.text
            error = parse_error_response(service_name, response.status_code, body)

            if response.status_code in transient_codes:
                last_error = error
                if attempt < max_retries:
                    delay = _get_retry_delay(response, base_delay, attempt, max_delay)
                    time.sleep(delay)
                    continue
                raise last_error
            raise error

        except (
            ReqConnectionError,
            ReqTimeout,
            ReqChunkedEncodingError,
        ) as exc:
            last_error = APITransientError(
                service_name, 0, f"Connection error: {exc}"
            )
            if attempt < max_retries:
                delay = min(max_delay, base_delay * (2 ** attempt))
                # Add jitter (±25%) to prevent retry storms
                delay *= 0.75 + random.random() * 0.5
                time.sleep(delay)
                continue
            raise last_error from exc

    # Defensive: should be unreachable; outer raise covers it.
    if last_error:
        raise last_error
    raise APITransientError(service_name, 0, "Retry loop exited without response")


def _get_retry_delay(
    response: requests.Response,
    base_delay: float,
    attempt: int,
    max_delay: float = 60.0,
) -> float:
    """Calculate retry delay, honoring Retry-After header if present.

    Supports both `delta-seconds` (e.g. "120") and HTTP-date formats
    per RFC 7231 §7.1.3. Falls back to exponential backoff with jitter.
    Clamped to [0, max_delay] to prevent indefinite stalls from a
    misbehaving server.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        retry_after = retry_after.strip()
        # Try delta-seconds first
        try:
            secs = float(retry_after)
            return max(0.0, min(secs, max_delay))
        except ValueError:
            pass
        # Try HTTP-date
        try:
            target = parsedate_to_datetime(retry_after)
            if target is not None:
                import datetime
                now = datetime.datetime.now(target.tzinfo or datetime.timezone.utc)
                delta = (target - now).total_seconds()
                if delta > 0:
                    return min(delta, max_delay)
        except (TypeError, ValueError):
            pass
    delay = min(max_delay, base_delay * (2 ** attempt))
    # ±25% jitter to spread out retries from concurrent callers
    delay *= 0.75 + random.random() * 0.5
    return delay


def download_file(
    url: str,
    *,
    retries: int = 3,
    timeout: float = 120,
    chunk_size: int = 8192,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    session: Optional[requests.Session] = None,
) -> bytes:
    """Download a file with retry logic and size cap.

    Raises APIPermanentError if the response exceeds `max_bytes` (DoS guard).
    Streams chunks with iter_content so we abort early instead of buffering
    the entire body in memory before checking length.
    """
    last_error = None
    requester = session or requests

    for attempt in range(retries + 1):
        try:
            response = requester.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Respect Content-Length if present
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        raise APIPermanentError(
                            "download", 413,
                            f"Response Content-Length {content_length} exceeds cap {max_bytes}",
                        )
                except ValueError:
                    pass  # Malformed header — fall through to streaming check

            buffer = io.BytesIO()
            total = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise APIPermanentError(
                        "download", 413,
                        f"Streamed response exceeded cap {max_bytes} bytes",
                    )
                buffer.write(chunk)
            return buffer.getvalue()

        except APIPermanentError:
            # Don't retry size-cap or 4xx errors
            raise
        except ReqRequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * (2 ** attempt) * (0.75 + random.random() * 0.5))
                continue

    raise APITransientError(
        "download", 0, f"Download failed after {retries + 1} attempts: {last_error}"
    )
