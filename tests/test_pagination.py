"""Voice pagination termination tests — guard against infinite loops."""
from unittest.mock import MagicMock, patch

import pytest

import utils


@pytest.fixture(autouse=True)
def _clear_cache():
    utils._voice_cache.clear()
    utils._voice_cache_time.clear()
    yield
    utils._voice_cache.clear()
    utils._voice_cache_time.clear()


def _voice_page(voices, has_more=False, next_token=None):
    resp = MagicMock()
    resp.json = MagicMock(return_value={
        "voices": voices,
        "has_more": has_more,
        "next_page_token": next_token,
    })
    return resp


def test_pagination_terminates_when_has_more_false():
    page1 = _voice_page([{"voice_id": "a", "labels": {}}], has_more=False)
    with patch("utils.api_get", return_value=page1):
        result = utils.fetch_all_voices("test_key")
    assert len(result) == 1


def test_pagination_terminates_when_has_more_true_but_no_token():
    """API bug: has_more=True but next_page_token missing → must NOT loop forever."""
    page1 = _voice_page([{"voice_id": "a", "labels": {}}], has_more=True, next_token=None)
    with patch("utils.api_get", return_value=page1):
        result = utils.fetch_all_voices("test_key")
    assert len(result) == 1  # Stopped after the first page despite has_more=True


def test_pagination_follows_next_page_token():
    page1 = _voice_page(
        [{"voice_id": "a", "labels": {}}],
        has_more=True, next_token="TOKEN_2",
    )
    page2 = _voice_page(
        [{"voice_id": "b", "labels": {}}],
        has_more=False,
    )
    api_get_mock = MagicMock(side_effect=[page1, page2])
    with patch("utils.api_get", api_get_mock):
        result = utils.fetch_all_voices("test_key")
    assert len(result) == 2
    assert {v["voice_id"] for v in result} == {"a", "b"}
    assert api_get_mock.call_count == 2


def test_pagination_caps_at_max_pages():
    """Even if the API keeps saying has_more=True with valid tokens, we cap."""
    # Always say has_more=True with a fresh token — infinite without the cap.
    counter = {"n": 0}
    def make_page(*a, **kw):
        counter["n"] += 1
        return _voice_page([{"voice_id": f"v{counter['n']}", "labels": {}}],
                           has_more=True, next_token=f"tok_{counter['n']}")
    with patch("utils.api_get", side_effect=make_page):
        result = utils.fetch_all_voices("test_key")
    # Default max_pages = 100; verify we don't blow past it
    assert len(result) <= 105


def test_voice_cache_uses_cache_on_second_call():
    page = _voice_page([{"voice_id": "x", "labels": {}}], has_more=False)
    api_get_mock = MagicMock(return_value=page)
    with patch("utils.api_get", api_get_mock):
        utils.fetch_all_voices("test_key")
        utils.fetch_all_voices("test_key")  # second call hits cache
    assert api_get_mock.call_count == 1


def test_voice_cache_force_refresh_bypasses_cache():
    page = _voice_page([{"voice_id": "x", "labels": {}}], has_more=False)
    api_get_mock = MagicMock(return_value=page)
    with patch("utils.api_get", api_get_mock):
        utils.fetch_all_voices("test_key")
        utils.fetch_all_voices("test_key", force_refresh=True)
    assert api_get_mock.call_count == 2


def test_voice_cache_bounded_size():
    """Cache should not grow unboundedly across many distinct keys."""
    page = _voice_page([{"voice_id": "x", "labels": {}}], has_more=False)
    with patch("utils.api_get", return_value=page):
        for i in range(50):  # > _VOICE_CACHE_MAX_KEYS
            utils.fetch_all_voices(f"key_{i}")
    assert len(utils._voice_cache) <= utils._VOICE_CACHE_MAX_KEYS
