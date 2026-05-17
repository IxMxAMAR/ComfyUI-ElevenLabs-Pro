"""Secret-leakage regression + error parsing tests."""
import json

import pytest

from shared.errors import (
    APIError,
    APIPermanentError,
    APITransientError,
    APIQuotaError,
    parse_error_response,
    scrub_secrets,
)


# --------- scrub_secrets ---------

def test_scrub_secrets_redacts_explicit_token():
    msg = "ConnectionError: failed to reach api with key sk_abc1234567890xyz"
    out = scrub_secrets(msg, extra_secrets=["sk_abc1234567890xyz"])
    assert "sk_abc1234567890xyz" not in out
    assert "***REDACTED***" in out


def test_scrub_secrets_redacts_url_query_param():
    msg = "GET https://api.example.com/v1/voices?xi-api-key=SECRETabcdef1234567890 -> 401"
    out = scrub_secrets(msg)
    assert "SECRETabcdef1234567890" not in out
    assert "REDACTED" in out


def test_scrub_secrets_url_pattern_with_token_param():
    msg = "url=https://x.com/y?token=abc1234567890123456&z=1"
    out = scrub_secrets(msg)
    assert "abc1234567890123456" not in out


def test_scrub_secrets_handles_empty():
    assert scrub_secrets("") == ""
    assert scrub_secrets(None) is None


def test_scrub_secrets_ignores_short_secrets():
    msg = "Used key=AAAA in test"
    out = scrub_secrets(msg, extra_secrets=["AAAA"])
    assert "AAAA" in out


# --------- parse_error_response ---------

def test_parse_error_response_json_detail_message():
    body = json.dumps({"detail": {"message": "Quota exceeded"}})
    err = parse_error_response("ElevenLabs", 402, body)
    assert isinstance(err, APIQuotaError)
    assert "Quota exceeded" in err.detail


def test_parse_error_response_json_detail_string():
    body = json.dumps({"detail": "Bad voice id"})
    err = parse_error_response("ElevenLabs", 422, body)
    assert isinstance(err, APIPermanentError)
    assert err.detail == "Bad voice id"


def test_parse_error_response_429_is_transient():
    body = json.dumps({"detail": "Too many requests"})
    err = parse_error_response("ElevenLabs", 429, body)
    assert isinstance(err, APITransientError)


def test_parse_error_response_500_is_transient():
    err = parse_error_response("ElevenLabs", 502, "Bad Gateway")
    assert isinstance(err, APITransientError)


def test_parse_error_response_secret_scrubbed():
    key = "sk_abc123def456ghi789jkl012"
    body = json.dumps({"detail": "Invalid key " + key})
    err = parse_error_response("ElevenLabs", 401, body, extra_secrets=[key])
    assert key not in err.detail
    assert "REDACTED" in err.detail
    assert key not in str(err)


def test_parse_error_response_non_json_body_truncated():
    body = "x" * 10000
    err = parse_error_response("ElevenLabs", 500, body)
    assert len(err.detail) <= 510


def test_parse_error_response_handles_none_body():
    err = parse_error_response("ElevenLabs", 500, "")
    assert isinstance(err, APITransientError)


def test_parse_error_response_handles_binary_via_response_obj():
    class FakeResp:
        headers = {"Content-Type": "audio/mpeg"}
        text = "garbage-binary-text"
    err = parse_error_response("ElevenLabs", 500, FakeResp())
    assert "binary response" in err.detail.lower() or err.detail == "<binary response>"


# --------- APIError class ---------

def test_api_error_str_contains_service_and_status():
    err = APIError("X", 404, "not found")
    assert "[X]" in str(err)
    assert "404" in str(err)
    assert "not found" in str(err)


def test_quota_error_inherits_permanent():
    e = APIQuotaError("X", 402, "out of credits")
    assert isinstance(e, APIPermanentError)
