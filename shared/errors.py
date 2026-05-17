"""Standardized API error classes and response parsing for all services."""

import json
import re


class APIError(RuntimeError):
    """Base API error with service name, HTTP status code, and detail message."""

    def __init__(self, service: str, status_code: int, detail: str):
        self.service = service
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{service}] HTTP {status_code}: {detail}")


class APITransientError(APIError):
    """Retryable API error (e.g. 429, 500, 502, 503, 504)."""
    pass


class APIPermanentError(APIError):
    """Non-retryable API error (e.g. 400, 401, 403, 404)."""
    pass


class APIQuotaError(APIPermanentError):
    """Credits or quota exhausted."""
    pass


_MAX_DETAIL_LEN = 500
# Pattern for ElevenLabs API keys: typically `xi_` or hex of length ~40-48,
# but we also redact anything that looks like a high-entropy alnum token >=20 chars
# in case a key flows through. Conservative: only redact tokens that look key-ish.
_KEY_PAT = re.compile(r"(?i)((?:xi[_-])?(?:api[_-])?key[=:\s\"\']\s*)([A-Za-z0-9_\-\.]{20,})")
_URL_PARAM_PAT = re.compile(r"(?i)(\?|&)(xi[-_]api[-_]key|api[-_]key|key|token)=([^&\s\"\']+)")


def _safe_body(resp_or_body) -> str:
    """Extract a safe textual body from a response or raw body.

    Returns a short placeholder for binary / huge / undecodable bodies so we
    never echo an entire 5MB binary stream into an exception message.
    """
    # Accept either a string or a requests.Response
    if isinstance(resp_or_body, str):
        body = resp_or_body
    else:
        try:
            ct = resp_or_body.headers.get("Content-Type", "") if hasattr(resp_or_body, "headers") else ""
            ct_l = ct.lower()
            if any(t in ct_l for t in ("audio/", "video/", "image/", "octet-stream")):
                return "<binary response>"
            body = resp_or_body.text
        except Exception:
            return "<unreadable response>"
    if body is None:
        return ""
    if len(body) > 4096:
        body = body[:4096] + "...<truncated>"
    return body


def scrub_secrets(text: str, extra_secrets=None) -> str:
    """Best-effort scrub of API keys / tokens from a string.

    1. Replaces any explicit `extra_secrets` strings (e.g. the user's key
       passed by caller) wholesale.
    2. Redacts `?xi-api-key=...`, `&api_key=...` query-string params.
    3. Redacts key-like patterns near `key:` / `api_key=` markers.

    NOT cryptographically rigorous — defense in depth, not a perfect filter.
    """
    if not text:
        return text
    out = text
    if extra_secrets:
        for s in extra_secrets:
            if s and isinstance(s, str) and len(s) >= 8:
                out = out.replace(s, "***REDACTED***")
    out = _URL_PARAM_PAT.sub(lambda m: f"{m.group(1)}{m.group(2)}=***REDACTED***", out)
    out = _KEY_PAT.sub(lambda m: f"{m.group(1)}***REDACTED***", out)
    return out


def parse_error_response(service: str, status_code: int, body, extra_secrets=None) -> APIError:
    """Parse an error response body and return the appropriate APIError subclass.

    `body` may be a string OR a `requests.Response` — Content-Type sniffing
    avoids decoding binary payloads as text. `extra_secrets` (e.g. the API
    key) is scrubbed from the detail message before construction so user
    secrets cannot leak into ComfyUI logs / UI / saved workflow PNGs.
    """
    body_str = _safe_body(body)
    detail = ""

    try:
        data = json.loads(body_str)

        if isinstance(data, dict) and "detail" in data:
            d = data["detail"]
            if isinstance(d, dict) and "message" in d:
                detail = d["message"]
            elif isinstance(d, str):
                detail = d
            else:
                detail = str(d)
        elif isinstance(data, dict) and "message" in data:
            detail = data["message"]
        else:
            detail = str(data)

    except (json.JSONDecodeError, ValueError, TypeError):
        detail = body_str

    if not isinstance(detail, str):
        detail = str(detail)

    detail = scrub_secrets(detail, extra_secrets=extra_secrets)

    if len(detail) > _MAX_DETAIL_LEN:
        detail = detail[:_MAX_DETAIL_LEN] + "..."

    if status_code == 429 or status_code >= 500:
        return APITransientError(service, status_code, detail)
    elif status_code in (402, 403) and any(
        kw in detail.lower() for kw in ("quota", "credit", "limit", "exhausted")
    ):
        return APIQuotaError(service, status_code, detail)
    else:
        return APIPermanentError(service, status_code, detail)
