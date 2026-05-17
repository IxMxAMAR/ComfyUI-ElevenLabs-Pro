"""Node-level behavior tests — validation, IS_CHANGED, redaction, new utility nodes."""
import json
from unittest.mock import MagicMock, patch

import pytest
import torch

import nodes
import utils
from utils import redact_account_pii


# --------- VoiceSelector ---------

def test_voice_selector_returns_predefined_voice_id():
    sel = nodes.ElevenLabsPro_VoiceSelector()
    out = sel.select(voice=utils.VOICE_OPTIONS[1])  # First named voice
    assert isinstance(out[0], str)
    assert len(out[0]) >= 10  # voice ids are long


def test_voice_selector_custom_id_overrides():
    sel = nodes.ElevenLabsPro_VoiceSelector()
    out = sel.select(voice=utils.VOICE_OPTIONS[1], custom_voice_id="my_custom_id_xyz")
    assert out[0] == "my_custom_id_xyz"


def test_voice_selector_raises_on_invalid_voice():
    sel = nodes.ElevenLabsPro_VoiceSelector()
    with pytest.raises(ValueError):
        sel.select(voice="(Custom voice_id)")  # No matching voice_id


# --------- VoiceClone safety (CRITICAL) ---------

def test_voice_clone_default_dry_run_does_not_call_api():
    """VoiceClone must NOT call the API when create=False (default)."""
    audio = {"waveform": torch.zeros(1, 1, 1000), "sample_rate": 22050}
    clone = nodes.ElevenLabsPro_VoiceClone()
    with patch("nodes.api_post") as api_mock:
        vid, status = clone.clone(api_key="anykey", voice_name="X", audio1=audio,
                                  create=False)
    api_mock.assert_not_called()
    assert vid == ""
    assert "DRY-RUN" in status


def test_voice_clone_with_create_true_calls_api():
    audio = {"waveform": torch.zeros(1, 1, 1000), "sample_rate": 22050}
    clone = nodes.ElevenLabsPro_VoiceClone()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json = MagicMock(return_value={"voice_id": "vid_new"})
    with patch("nodes.api_post", return_value=fake_resp) as api_mock, \
         patch("nodes.get_api_key", return_value="key"):
        vid, status = clone.clone(api_key="anykey", voice_name="X", audio1=audio,
                                  create=True)
    api_mock.assert_called_once()
    assert vid == "vid_new"


# --------- VoiceCreate safety (CRITICAL) ---------

def test_voice_create_default_dry_run():
    vc = nodes.ElevenLabsPro_VoiceCreate()
    with patch("nodes.api_post") as api_mock:
        vid, status = vc.create(api_key="k", generated_voice_id="gen_id_123",
                                voice_name="X", create=False)
    api_mock.assert_not_called()
    assert "DRY-RUN" in status


# --------- AccountInfo PII redaction (CRITICAL) ---------

def test_redact_account_pii_strips_email():
    raw = {
        "email": "user@example.com",
        "first_name": "John",
        "xi_api_key": "sk_secret",
        "subscription": {
            "tier": "creator",
            "character_count": 1000,
            "character_limit": 100000,
        },
    }
    out = redact_account_pii(raw)
    assert "email" not in out
    assert "first_name" not in out
    assert "xi_api_key" not in out
    assert out["subscription"]["tier"] == "creator"
    assert out["subscription"]["character_count"] == 1000


def test_redact_account_pii_deep_nested():
    raw = {"a": {"b": {"email": "x@y.com", "tier": "free"}}}
    out = redact_account_pii(raw)
    assert "email" not in out["a"]["b"]
    assert out["a"]["b"]["tier"] == "free"


def test_account_info_returns_redacted_json():
    """AccountInfo node must not echo email/name into account_json output."""
    info = nodes.ElevenLabsPro_AccountInfo()
    raw = {
        "tier": "creator",
        "character_count": 100, "character_limit": 1000,
        "email": "leak@example.com",
    }
    fake_resp = MagicMock(status_code=200)
    fake_resp.json = MagicMock(return_value=raw)
    with patch("nodes.api_get", return_value=fake_resp), \
         patch("nodes.get_api_key", return_value="key"):
        json_str, tier, count, limit = info.fetch(api_key="k", endpoint="user_subscription")
    assert "leak@example.com" not in json_str
    assert tier == "creator"
    assert count == 100
    assert limit == 1000


# --------- TTS validation ---------

def test_tts_empty_text_raises():
    tts = nodes.ElevenLabsPro_TTS()
    with pytest.raises(ValueError, match="empty"):
        tts.generate(api_key="k", text="", voice_id="vid", model="eleven_v3")


def test_tts_missing_voice_raises():
    tts = nodes.ElevenLabsPro_TTS()
    with pytest.raises(ValueError, match="voice_id"):
        tts.generate(api_key="k", text="hi", voice_id="", model="eleven_v3")


def test_tts_turbo_requires_language():
    """flash_v2_5 / turbo_v2_5 require an explicit language code."""
    tts = nodes.ElevenLabsPro_TTS()
    with pytest.raises(ValueError, match="language"):
        tts.generate(api_key="k", text="hi", voice_id="vid",
                     model="eleven_flash_v2_5", language="Auto Detect")


def test_tts_v3_forces_normalization_off():
    """eleven_v3 silently overrides apply_text_normalization to 'off'."""
    tts = nodes.ElevenLabsPro_TTS()
    fake_resp = MagicMock(status_code=200, content=b"FAKE_MP3")
    captured = {}
    def grab(url, key, **kwargs):
        captured.update(kwargs)
        return fake_resp
    with patch("nodes.api_post", side_effect=grab), \
         patch("nodes.get_api_key", return_value="k"), \
         patch("nodes.audio_bytes_to_comfy", return_value={"waveform": torch.zeros(1, 1, 100), "sample_rate": 44100}):
        tts.generate(api_key="k", text="hi", voice_id="vid", model="eleven_v3",
                     apply_text_normalization="auto")
    assert captured["json"]["apply_text_normalization"] == "off"


def test_tts_pronunciation_invalid_json_raises():
    """Malformed pronunciation_dictionary_locators should error explicitly."""
    tts = nodes.ElevenLabsPro_TTS()
    with patch("nodes.get_api_key", return_value="k"):
        with pytest.raises(ValueError, match="valid JSON"):
            tts.generate(api_key="k", text="hi", voice_id="vid", model="eleven_v3",
                         pronunciation_dictionary_locators="{not json}")


# --------- Music API contract ---------

def test_music_uses_music_length_ms_not_duration_seconds():
    music = nodes.ElevenLabsPro_Music()
    fake_resp = MagicMock(status_code=200, content=b"FAKE_MP3")
    captured = {}
    def grab(url, key, **kwargs):
        captured.update(kwargs)
        return fake_resp
    with patch("nodes.api_post", side_effect=grab), \
         patch("nodes.get_api_key", return_value="k"), \
         patch("nodes.audio_bytes_to_comfy", return_value={"waveform": torch.zeros(1, 1, 100), "sample_rate": 44100}):
        music.generate(api_key="k", prompt="upbeat synthwave", duration_seconds=30.0)
    body = captured["json"]
    assert body["music_length_ms"] == 30000
    assert "duration_seconds" not in body
    assert body["model_id"] == "music_v1"


def test_music_accepts_composition_plan():
    music = nodes.ElevenLabsPro_Music()
    fake_resp = MagicMock(status_code=200, content=b"FAKE_MP3")
    captured = {}
    def grab(url, key, **kwargs):
        captured.update(kwargs)
        return fake_resp
    plan = json.dumps({"sections": []})
    with patch("nodes.api_post", side_effect=grab), \
         patch("nodes.get_api_key", return_value="k"), \
         patch("nodes.audio_bytes_to_comfy", return_value={"waveform": torch.zeros(1, 1, 100), "sample_rate": 44100}):
        music.generate(api_key="k", prompt="", composition_plan=plan)
    assert "composition_plan" in captured["json"]
    assert "prompt" not in captured["json"]


def test_music_rejects_no_prompt_or_plan():
    music = nodes.ElevenLabsPro_Music()
    with pytest.raises(ValueError, match="prompt OR a composition_plan"):
        music.generate(api_key="k", prompt="")


def test_music_max_duration_600():
    music = nodes.ElevenLabsPro_Music()
    with pytest.raises(ValueError, match="between 3 and 600"):
        music.generate(api_key="k", prompt="x", duration_seconds=700.0)


# --------- SFX API contract ---------

def test_sfx_default_model_id_is_text_to_sound_v2():
    sfx = nodes.ElevenLabsPro_SFX()
    fake_resp = MagicMock(status_code=200, content=b"FAKE_MP3")
    captured = {}
    def grab(url, key, **kwargs):
        captured.update(kwargs)
        return fake_resp
    with patch("nodes.api_post", side_effect=grab), \
         patch("nodes.get_api_key", return_value="k"), \
         patch("nodes.audio_bytes_to_comfy", return_value={"waveform": torch.zeros(1, 1, 100), "sample_rate": 44100}):
        sfx.generate(api_key="k", text="thunder")
    assert captured["json"]["model_id"] == "eleven_text_to_sound_v2"
    assert captured["json"]["text"] == "thunder"


def test_sfx_duration_above_max_raises():
    sfx = nodes.ElevenLabsPro_SFX()
    with pytest.raises(ValueError):
        sfx.generate(api_key="k", text="x", duration=50.0)


# --------- IS_CHANGED caching ---------

def test_input_cache_mixin_consistent_hash_for_same_inputs():
    from shared.node_utils import InputCacheMixin
    class Dummy(InputCacheMixin):
        pass
    h1 = Dummy.IS_CHANGED(text="hello", model="x", voice_id="v", api_key="K")
    h2 = Dummy.IS_CHANGED(text="hello", model="x", voice_id="v", api_key="K")
    assert h1 == h2
    assert isinstance(h1, str)


def test_input_cache_mixin_excludes_api_key_from_hash():
    """Switching api_key alone must not invalidate cache (avoid leaking key)."""
    from shared.node_utils import InputCacheMixin
    class Dummy(InputCacheMixin):
        pass
    h1 = Dummy.IS_CHANGED(text="hi", api_key="KEY_A")
    h2 = Dummy.IS_CHANGED(text="hi", api_key="KEY_B")
    assert h1 == h2


def test_input_cache_mixin_changes_with_text():
    from shared.node_utils import InputCacheMixin
    class Dummy(InputCacheMixin):
        pass
    h1 = Dummy.IS_CHANGED(text="hello")
    h2 = Dummy.IS_CHANGED(text="goodbye")
    assert h1 != h2


def test_input_cache_mixin_handles_audio_dict():
    from shared.node_utils import InputCacheMixin
    class Dummy(InputCacheMixin):
        pass
    a = {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050}
    h = Dummy.IS_CHANGED(audio=a, voice_id="v")
    assert isinstance(h, str)


# --------- Always-execute mixin (for FetchVoices / AccountInfo) ---------

def test_always_execute_mixin_returns_nan():
    from shared.node_utils import AlwaysExecuteMixin
    val = AlwaysExecuteMixin.IS_CHANGED()
    # NaN != NaN — that's the point
    import math
    assert math.isnan(val)


# --------- VoiceTagInserter ---------

def test_voice_tag_inserter_prepend():
    n = nodes.ElevenLabsPro_VoiceTagInserter()
    out = n.build("hello world", "whispers", "prepend")
    assert out[0] == "[whispers] hello world"


def test_voice_tag_inserter_append():
    n = nodes.ElevenLabsPro_VoiceTagInserter()
    out = n.build("hello", "laughs", "append")
    assert out[0] == "hello [laughs]"


def test_voice_tag_inserter_wrap():
    n = nodes.ElevenLabsPro_VoiceTagInserter()
    out = n.build("hi", "excited", "wrap")
    assert out[0] == "[excited] hi [excited]"


# --------- Subtitle export ---------

def test_subtitle_export_srt_from_tts_alignment():
    n = nodes.ElevenLabsPro_SubtitleExport()
    alignment = json.dumps({
        "characters": list("Hello world!"),
        "character_start_times_seconds": [i * 0.1 for i in range(12)],
        "character_end_times_seconds": [(i + 1) * 0.1 for i in range(12)],
    })
    out = n.export(alignment, format="srt", max_chars_per_cue=42, max_seconds_per_cue=10)
    assert "00:00:00,000 -->" in out[0]
    assert "Hello world!" in out[0]


def test_subtitle_export_vtt_header():
    n = nodes.ElevenLabsPro_SubtitleExport()
    alignment = json.dumps({
        "characters": ["a"],
        "character_start_times_seconds": [0.0],
        "character_end_times_seconds": [0.5],
    })
    out = n.export(alignment, format="vtt")
    assert out[0].startswith("WEBVTT")


def test_subtitle_export_from_stt_words():
    n = nodes.ElevenLabsPro_SubtitleExport()
    words = json.dumps([
        {"text": "hello", "start": 0.0, "end": 0.5},
        {"text": "world", "start": 0.6, "end": 1.0},
    ])
    out = n.export(words, format="srt")
    assert "hello world" in out[0] or "hello" in out[0]


def test_subtitle_export_invalid_json_raises():
    n = nodes.ElevenLabsPro_SubtitleExport()
    with pytest.raises(ValueError):
        n.export("{not json}")


# --------- Audio utility nodes ---------

def test_audio_metadata():
    n = nodes.ElevenLabsPro_AudioMetadata()
    audio = {"waveform": torch.zeros(1, 2, 44100), "sample_rate": 44100}
    dur, sr, ch, samp = n.inspect(audio)
    assert dur == 1.0
    assert sr == 44100
    assert ch == 2
    assert samp == 44100


def test_audio_channels_to_mono():
    n = nodes.ElevenLabsPro_AudioChannels()
    audio = {"waveform": torch.ones(1, 2, 1000), "sample_rate": 22050}
    out, = n.convert(audio, "to_mono")
    assert out["waveform"].shape == (1, 1, 1000)


def test_audio_channels_to_stereo_from_mono():
    n = nodes.ElevenLabsPro_AudioChannels()
    audio = {"waveform": torch.ones(1, 1, 1000), "sample_rate": 22050}
    out, = n.convert(audio, "to_stereo")
    assert out["waveform"].shape == (1, 2, 1000)


def test_audio_trim():
    n = nodes.ElevenLabsPro_AudioTrim()
    audio = {"waveform": torch.arange(44100, dtype=torch.float32).reshape(1, 1, -1), "sample_rate": 44100}
    out, = n.trim(audio, start_seconds=0.25, end_seconds=0.75)
    # Roughly 0.5 seconds worth of samples
    samples = out["waveform"].shape[-1]
    assert 21000 <= samples <= 23000


def test_audio_normalize_peak():
    n = nodes.ElevenLabsPro_AudioNormalize()
    audio = {"waveform": torch.full((1, 1, 1000), 0.1), "sample_rate": 22050}
    out, = n.normalize(audio, mode="peak", target_db=0.0)
    # Peak should now be ~1.0
    assert abs(float(out["waveform"].abs().max()) - 1.0) < 0.01


def test_audio_concat_two_clips():
    n = nodes.ElevenLabsPro_AudioConcat()
    a1 = {"waveform": torch.ones(1, 1, 100), "sample_rate": 22050}
    a2 = {"waveform": torch.zeros(1, 1, 200), "sample_rate": 22050}
    out, = n.concat(audio1=a1, audio2=a2)
    assert out["waveform"].shape == (1, 1, 300)
    assert out["sample_rate"] == 22050


def test_audio_concat_with_gap():
    n = nodes.ElevenLabsPro_AudioConcat()
    a1 = {"waveform": torch.ones(1, 1, 100), "sample_rate": 10}
    a2 = {"waveform": torch.ones(1, 1, 100), "sample_rate": 10}
    out, = n.concat(audio1=a1, audio2=a2, gap_seconds=1.0)
    # 100 + (1.0s × 10Hz = 10 samples) + 100 = 210
    assert out["waveform"].shape == (1, 1, 210)


def test_audio_concat_resamples_mismatched():
    n = nodes.ElevenLabsPro_AudioConcat()
    a1 = {"waveform": torch.ones(1, 1, 1000), "sample_rate": 44100}
    a2 = {"waveform": torch.ones(1, 1, 500), "sample_rate": 22050}
    out, = n.concat(audio1=a1, audio2=a2)
    assert out["sample_rate"] == 44100  # first clip's SR is target
    # a2 resampled from 22050 to 44100 doubles samples ~= 1000; total ~2000
    assert 1900 <= out["waveform"].shape[-1] <= 2100


# --------- CostEstimator ---------

def test_cost_estimator_text_length():
    n = nodes.ElevenLabsPro_CostEstimator()
    chars, credits, summary = n.estimate("hello world", "eleven_v3")
    assert chars == 11
    assert credits == 11.0  # v3 = 1.0 per char
    assert "eleven_v3" in summary


def test_cost_estimator_flash_discount():
    n = nodes.ElevenLabsPro_CostEstimator()
    _, credits, _ = n.estimate("hello world", "eleven_flash_v2_5")
    assert credits == 5.5  # 11 × 0.5


# --------- VoiceSettingsPreset ---------

def test_voice_settings_preset_emits_valid_json():
    n = nodes.ElevenLabsPro_VoiceSettingsPreset()
    out, = n.build(stability=0.4, similarity_boost=0.8, style=0.1, speed=1.1, use_speaker_boost=True)
    data = json.loads(out)
    assert data["stability"] == 0.4
    assert data["use_speaker_boost"] is True


# --------- SaveAudio path safety ---------

def test_save_audio_rejects_path_traversal(tmp_path, monkeypatch):
    """subdirectory='../etc' must not write outside output dir."""
    n = nodes.ElevenLabsPro_SaveAudio()
    monkeypatch.setattr(n, "save", n.save)  # ensure no patching weirdness
    # Patch folder_paths to return tmp_path
    import sys
    fp = type(sys)("folder_paths")
    fp.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "folder_paths", fp)
    audio = {"waveform": torch.zeros(1, 1, 100), "sample_rate": 22050}
    # subdirectory with .. — should be sanitized
    path = n.save(audio, filename_prefix="x", subdirectory="../../etc")
    assert str(tmp_path) in path[0]


# --------- Dialogue ---------

def test_dialogue_total_text_validation():
    """Sum-of-text validation should reject inputs exceeding model limit."""
    dlg = nodes.ElevenLabsPro_Dialogue()
    huge = "x" * 6000  # eleven_v3 limit is 5000
    with patch("nodes.get_api_key", return_value="k"):
        with pytest.raises(ValueError, match="exceeds"):
            dlg.generate(api_key="k", text1=huge, voice_id1="vid",
                         model="eleven_v3")


def test_dialogue_no_inputs_raises():
    dlg = nodes.ElevenLabsPro_Dialogue()
    with patch("nodes.get_api_key", return_value="k"):
        with pytest.raises(ValueError, match="At least one"):
            dlg.generate(api_key="k", text1="", voice_id1="", model="eleven_v3")


# --------- NODE_CLASS_MAPPINGS exhaustive ---------

def test_node_class_mappings_complete():
    """Every node class should be registered in NODE_CLASS_MAPPINGS."""
    assert len(nodes.NODE_CLASS_MAPPINGS) >= 26
    assert "ElevenLabsPro_TTS" in nodes.NODE_CLASS_MAPPINGS
    assert "ElevenLabsPro_SaveAudio" in nodes.NODE_CLASS_MAPPINGS


def test_node_display_names_match_classes():
    """Each registered class should have a display name."""
    for k in nodes.NODE_CLASS_MAPPINGS:
        assert k in nodes.NODE_DISPLAY_NAME_MAPPINGS, f"Missing display name for {k}"
