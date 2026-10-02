"""Request-shape tests for current ElevenLabs endpoints, models and parameters. HTTP is mocked."""
import base64
import json
from unittest.mock import MagicMock, patch

import pytest
import torch

import nodes
import utils

AUDIO = {"waveform": torch.zeros(1, 1, 1000), "sample_rate": 22050}
FAKE_DECODED = {"waveform": torch.zeros(1, 1, 100), "sample_rate": 44100}


class Capture:
    """Patches api_post/get_api_key/audio decoding and records the last request."""

    def __init__(self, json_payload=None, content=b"FAKE"):
        self.url = None
        self.kwargs = None
        self.decoded_format = None
        resp = MagicMock(status_code=200, content=content)
        resp.json = MagicMock(return_value=json_payload if json_payload is not None else {})
        self.resp = resp

    def _post(self, url, key, **kwargs):
        self.url = url
        self.kwargs = kwargs
        return self.resp

    def _decode(self, audio_bytes, fmt):
        self.decoded_format = fmt
        return FAKE_DECODED

    def __enter__(self):
        self._patches = [
            patch("nodes.api_post", side_effect=self._post),
            patch("nodes.get_api_key", return_value="k"),
            patch("nodes.audio_bytes_to_comfy", side_effect=self._decode),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()

    @property
    def body(self):
        return self.kwargs["json"]

    @property
    def form(self):
        return self.kwargs["data"]


def _optional_names(node_cls):
    return list(node_cls.INPUT_TYPES()["optional"])


# --------- Models: shut down, new, limits ---------

def test_shut_down_models_removed():
    assert "eleven_monolingual_v1" not in utils.TTS_MODELS
    assert "eleven_multilingual_v1" not in utils.TTS_MODELS
    assert "scribe_v1" not in utils.STT_MODELS
    assert "eleven_monolingual_v1" not in nodes.ElevenLabsPro_CostEstimator._COST_PER_CHAR


def test_new_models_available():
    assert "eleven_v4" in utils.TTS_MODELS
    assert "eleven_v4" in utils.DIALOGUE_MODELS
    assert "scribe_v2_medical" in utils.STT_MODELS
    assert {"music_v1", "music_v2", "music_v2_5"} <= set(utils.MUSIC_MODELS)
    assert utils.VOICE_DESIGN_MODELS == ["eleven_multilingual_ttv_v2", "eleven_ttv_v3"]


def test_defaults_unchanged():
    assert nodes.ElevenLabsPro_TTS.INPUT_TYPES()["required"]["model"][1]["default"] == "eleven_v3"
    assert nodes.ElevenLabsPro_STT.INPUT_TYPES()["required"]["model"][1]["default"] == "scribe_v2"
    assert nodes.ElevenLabsPro_Music.INPUT_TYPES()["optional"]["model"][1]["default"] == "music_v1"
    assert nodes.ElevenLabsPro_Dialogue.INPUT_TYPES()["required"]["model"][1]["default"] == "eleven_v3"


def test_char_limits_follow_docs():
    utils.validate_text_length("x" * 10000, "eleven_v4")
    with pytest.raises(ValueError):
        utils.validate_text_length("x" * 10001, "eleven_v4")
    with pytest.raises(ValueError):
        utils.validate_text_length("x" * 30001, "eleven_flash_v2")


def test_tts_accepts_eleven_v4():
    with Capture() as cap:
        nodes.ElevenLabsPro_TTS().generate(api_key="k", text="hi", voice_id="vid", model="eleven_v4")
    assert cap.body["model_id"] == "eleven_v4"
    assert cap.url.endswith("/v1/text-to-speech/vid")


# --------- New widgets are appended after existing ones ---------

def test_new_inputs_are_appended_last():
    assert _optional_names(nodes.ElevenLabsPro_TTS)[-1] == "apply_language_text_normalization"
    assert _optional_names(nodes.ElevenLabsPro_TTSTimestamps)[:5] == [
        "stability", "similarity_boost", "output_format", "seed", "enable_logging",
    ]
    assert _optional_names(nodes.ElevenLabsPro_SFX)[-1] == "auto_duration"
    assert _optional_names(nodes.ElevenLabsPro_STT)[-3:] == [
        "transcript_edit", "use_multi_channel", "detect_speaker_roles",
    ]
    assert _optional_names(nodes.ElevenLabsPro_Music)[-5:] == [
        "finetune_id", "finetune_strength", "use_phonetic_names", "generation_mode", "lyrics_text",
    ]
    assert _optional_names(nodes.ElevenLabsPro_VoiceClone)[-1] == "labels"
    assert _optional_names(nodes.ElevenLabsPro_VoiceCreate) == ["voice_description", "labels"]
    assert _optional_names(nodes.ElevenLabsPro_VoiceDesign)[0] == "preview_index"
    assert _optional_names(nodes.ElevenLabsPro_Dialogue)[-4:] == [
        "previous_text", "future_text", "use_pvc_as_ivc", "pronunciation_dictionary_locators",
    ]
    assert nodes.ElevenLabsPro_STT.RETURN_NAMES[:3] == ("text", "language_code", "words_json")


def test_new_node_keys_registered():
    for key in ("ElevenLabsPro_VoiceRemix", "ElevenLabsPro_ForcedAlignment",
                "ElevenLabsPro_DialogueTimestamps", "ElevenLabsPro_MusicPlan"):
        assert key in nodes.NODE_CLASS_MAPPINGS
        assert key in nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert len(nodes.NODE_CLASS_MAPPINGS) == 30


# --------- Voice design / remix / create ---------

PREVIEWS = {
    "previews": [
        {"generated_voice_id": "gen1", "audio_base_64": base64.b64encode(b"mp3").decode()},
        {"generated_voice_id": "gen2", "audio_base_64": base64.b64encode(b"mp3").decode()},
    ],
    "text": "sample",
}
LONG_TEXT = "x" * 120


def test_voice_design_uses_design_endpoint():
    with Capture(PREVIEWS) as cap:
        audio, gen_id, all_ids = nodes.ElevenLabsPro_VoiceDesign().design(
            api_key="k", text=LONG_TEXT, voice_description="a warm baritone narrator voice",
            preview_index=1)
    assert cap.url.endswith("/v1/text-to-voice/design")
    assert cap.body == {"text": LONG_TEXT, "voice_description": "a warm baritone narrator voice"}
    assert gen_id == "gen2"
    assert all_ids == "gen1\ngen2"
    assert audio is FAKE_DECODED


def test_voice_design_short_text_is_auto_generated():
    with Capture(PREVIEWS) as cap:
        nodes.ElevenLabsPro_VoiceDesign().design(
            api_key="k", text="too short", voice_description="a warm baritone narrator voice")
    assert cap.body["auto_generate_text"] is True
    assert "text" not in cap.body


def test_voice_design_empty_text_raises_unless_auto():
    node = nodes.ElevenLabsPro_VoiceDesign()
    with pytest.raises(ValueError, match="Text is required"):
        node.design(api_key="k", text="", voice_description="a warm baritone narrator voice")
    with Capture(PREVIEWS) as cap:
        node.design(api_key="k", text="", voice_description="a warm baritone narrator voice",
                    auto_generate_text=True)
    assert cap.body["auto_generate_text"] is True


def test_voice_design_new_params():
    with Capture(PREVIEWS) as cap:
        nodes.ElevenLabsPro_VoiceDesign().design(
            api_key="k", text=LONG_TEXT, voice_description="a warm baritone narrator voice",
            model="eleven_ttv_v3", loudness=0.2, guidance_scale=7.0, seed=42,
            should_enhance=True, reference_audio=AUDIO, prompt_strength=0.8)
    body = cap.body
    assert body["model_id"] == "eleven_ttv_v3"
    assert body["loudness"] == 0.2
    assert body["guidance_scale"] == 7.0
    assert body["seed"] == 42
    assert body["should_enhance"] is True
    assert body["prompt_strength"] == 0.8
    assert base64.b64decode(body["reference_audio_base64"])[:4] == b"RIFF"


def test_voice_design_requires_description():
    with pytest.raises(ValueError, match="description"):
        nodes.ElevenLabsPro_VoiceDesign().design(api_key="k", text=LONG_TEXT, voice_description=" ")


def test_voice_design_no_previews_raises():
    with Capture({"previews": []}):
        with pytest.raises(RuntimeError, match="No voice previews"):
            nodes.ElevenLabsPro_VoiceDesign().design(
                api_key="k", text=LONG_TEXT, voice_description="a warm baritone narrator voice")


def test_voice_remix_request():
    with Capture(PREVIEWS) as cap:
        audio, gen_id, all_ids = nodes.ElevenLabsPro_VoiceRemix().remix(
            api_key="k", voice_id=" vid1 ", voice_description="make it older", text=LONG_TEXT,
            loudness=-0.5, guidance_scale=4.0, seed=7)
    assert cap.url.endswith("/v1/text-to-voice/vid1/remix")
    assert cap.body == {
        "text": LONG_TEXT, "voice_description": "make it older",
        "loudness": -0.5, "guidance_scale": 4.0, "seed": 7,
    }
    assert gen_id == "gen1"
    assert all_ids == "gen1\ngen2"


def test_voice_remix_default_guidance_not_sent():
    with Capture(PREVIEWS) as cap:
        nodes.ElevenLabsPro_VoiceRemix().remix(
            api_key="k", voice_id="v", voice_description="make it older", text=LONG_TEXT)
    assert "guidance_scale" not in cap.body


def test_voice_remix_requires_voice_id():
    with pytest.raises(ValueError, match="voice_id"):
        nodes.ElevenLabsPro_VoiceRemix().remix(
            api_key="k", voice_id="", voice_description="x" * 10, text=LONG_TEXT)


def test_voice_create_uses_text_to_voice_endpoint():
    with Capture({"voice_id": "vid_new"}) as cap:
        vid, status = nodes.ElevenLabsPro_VoiceCreate().create(
            api_key="k", generated_voice_id=" gen1 ", voice_name="N", create=True,
            voice_description="A calm, warm narrator voice for audiobooks",
            labels='{"accent": "british"}')
    assert cap.url.endswith("/v1/text-to-voice")
    assert cap.body == {
        "voice_name": "N",
        "voice_description": "A calm, warm narrator voice for audiobooks",
        "generated_voice_id": "gen1",
        "labels": {"accent": "british"},
    }
    assert vid == "vid_new"


def test_voice_create_short_description_raises():
    with patch("nodes.api_post") as api_mock:
        with pytest.raises(ValueError, match="20-1000"):
            nodes.ElevenLabsPro_VoiceCreate().create(
                api_key="k", generated_voice_id="g", voice_name="N", create=True,
                voice_description="too short")
    api_mock.assert_not_called()


def test_voice_create_dry_run_needs_no_description():
    with patch("nodes.api_post") as api_mock:
        _, status = nodes.ElevenLabsPro_VoiceCreate().create(
            api_key="k", generated_voice_id="g", voice_name="N", create=False)
    api_mock.assert_not_called()
    assert "DRY-RUN" in status


def test_labels_must_be_json_object():
    with pytest.raises(ValueError, match="valid JSON"):
        nodes._parse_labels("{nope")
    with pytest.raises(ValueError, match="JSON object"):
        nodes._parse_labels("[1]")
    assert nodes._parse_labels("") is None


def test_voice_clone_sends_labels():
    with Capture({"voice_id": "v"}) as cap:
        nodes.ElevenLabsPro_VoiceClone().clone(
            api_key="k", voice_name="X", audio1=AUDIO, create=True, labels='{"gender": "female"}')
    assert json.loads(cap.form["labels"]) == {"gender": "female"}


# --------- TTS / TTS with timestamps ---------

def test_tts_language_normalization_only_sent_when_enabled():
    with Capture() as cap:
        nodes.ElevenLabsPro_TTS().generate(api_key="k", text="hi", voice_id="v", model="eleven_v3")
    assert "apply_language_text_normalization" not in cap.body
    with Capture() as cap:
        nodes.ElevenLabsPro_TTS().generate(
            api_key="k", text="hi", voice_id="v", model="eleven_flash_v2_5",
            language="Japanese (ja)", apply_language_text_normalization=True)
    assert cap.body["apply_language_text_normalization"] is True


def test_tts_pronunciation_locators_still_sent():
    locators = [{"pronunciation_dictionary_id": "d1", "version_id": "v1"}]
    with Capture() as cap:
        nodes.ElevenLabsPro_TTS().generate(
            api_key="k", text="hi", voice_id="v", model="eleven_v3",
            pronunciation_dictionary_locators=json.dumps(locators))
    assert cap.body["pronunciation_dictionary_locators"] == locators


def test_tts_timestamps_decodes_audio_base64():
    payload = {
        "audio_base64": base64.b64encode(b"mp3bytes").decode(),
        "alignment": {"characters": ["h"], "character_start_times_seconds": [0.0],
                      "character_end_times_seconds": [0.1]},
    }
    with Capture(payload) as cap:
        audio, ts = nodes.ElevenLabsPro_TTSTimestamps().generate(
            api_key="k", text="hi", voice_id="v", model="eleven_v3")
    assert audio is FAKE_DECODED
    assert json.loads(ts)["characters"] == ["h"]
    assert cap.url.endswith("/v1/text-to-speech/v/with-timestamps")


def test_tts_timestamps_defaults_keep_request_unchanged():
    with Capture({"audio_base64": ""}) as cap:
        nodes.ElevenLabsPro_TTSTimestamps().generate(
            api_key="k", text="hi", voice_id="v", model="eleven_multilingual_v2")
    assert cap.body == {
        "text": "hi",
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        "apply_text_normalization": "auto",
    }


def test_tts_timestamps_new_params():
    with Capture({"audio_base64": ""}) as cap:
        nodes.ElevenLabsPro_TTSTimestamps().generate(
            api_key="k", text="hi", voice_id="v", model="eleven_v3",
            style=0.3, speed=1.2, use_speaker_boost=False, language="French (fr)",
            apply_text_normalization="on", previous_text=" before ", next_text=" after ")
    body = cap.body
    assert body["voice_settings"] == {
        "stability": 0.5, "similarity_boost": 0.75,
        "style": 0.3, "speed": 1.2, "use_speaker_boost": False,
    }
    assert body["language_code"] == "fr"
    assert body["apply_text_normalization"] == "off"   # eleven_v3 forces off
    assert body["previous_text"] == "before"
    assert body["next_text"] == "after"


# --------- SFX / audio isolation ---------

def test_sfx_auto_duration_omits_duration():
    with Capture() as cap:
        nodes.ElevenLabsPro_SFX().generate(api_key="k", text="thunder", auto_duration=True, duration=99.0)
    assert "duration_seconds" not in cap.body


def test_sfx_default_sends_duration():
    with Capture() as cap:
        nodes.ElevenLabsPro_SFX().generate(api_key="k", text="thunder")
    assert cap.body["duration_seconds"] == 5.0


def test_audio_isolation_sends_no_output_format_and_decodes_mp3():
    with Capture() as cap:
        nodes.ElevenLabsPro_AudioIsolation().isolate(api_key="k", audio=AUDIO, output_format="pcm_44100")
    assert "params" not in cap.kwargs
    assert cap.url.endswith("/v1/audio-isolation")
    assert cap.decoded_format == "mp3_44100_128"


# --------- Speech to text ---------

STT_RESPONSE = {"text": "hello", "language_code": "eng", "words": [{"text": "hello"}]}


def test_stt_sends_tag_audio_events_explicitly():
    with Capture(STT_RESPONSE) as cap:
        nodes.ElevenLabsPro_STT().transcribe(api_key="k", audio=AUDIO, model="scribe_v2")
    assert cap.form["tag_audio_events"] == "false"
    with Capture(STT_RESPONSE) as cap:
        nodes.ElevenLabsPro_STT().transcribe(
            api_key="k", audio=AUDIO, model="scribe_v2", tag_audio_events=True)
    assert cap.form["tag_audio_events"] == "true"


def test_stt_keyterms_sent_as_list():
    with Capture(STT_RESPONSE) as cap:
        nodes.ElevenLabsPro_STT().transcribe(
            api_key="k", audio=AUDIO, model="scribe_v2", keyterms=" ComfyUI, ElevenLabs ,, Scribe ")
    assert cap.form["keyterms"] == ["ComfyUI", "ElevenLabs", "Scribe"]


def test_stt_no_keyterms_not_sent():
    with Capture(STT_RESPONSE) as cap:
        nodes.ElevenLabsPro_STT().transcribe(api_key="k", audio=AUDIO, model="scribe_v2")
    assert "keyterms" not in cap.form
    for field in ("transcript_edit", "use_multi_channel", "multichannel_output_style",
                  "detect_speaker_roles"):
        assert field not in cap.form


def test_stt_medical_model_passthrough():
    with Capture(STT_RESPONSE) as cap:
        nodes.ElevenLabsPro_STT().transcribe(api_key="k", audio=AUDIO, model="scribe_v2_medical")
    assert cap.form["model_id"] == "scribe_v2_medical"


def test_stt_new_params_and_edited_text():
    response = dict(STT_RESPONSE, edited_transcript={"kind": "transcript", "text": "Hello."})
    with Capture(response) as cap:
        text, lang, words, edited = nodes.ElevenLabsPro_STT().transcribe(
            api_key="k", audio=AUDIO, model="scribe_v2", diarize=True,
            transcript_edit=" add punctuation ", use_multi_channel=True, detect_speaker_roles=True)
    assert cap.form["transcript_edit"] == "add punctuation"
    assert cap.form["use_multi_channel"] == "true"
    assert cap.form["multichannel_output_style"] == "combined"
    assert cap.form["detect_speaker_roles"] == "true"
    assert (text, lang, edited) == ("hello", "eng", "Hello.")
    assert json.loads(words) == [{"text": "hello"}]


def test_stt_edit_error_returns_empty_edited_text():
    response = dict(STT_RESPONSE, edited_transcript={
        "kind": "error", "error_type": "edit_failed", "message": "could not edit"})
    with Capture(response):
        out = nodes.ElevenLabsPro_STT().transcribe(
            api_key="k", audio=AUDIO, model="scribe_v2", transcript_edit="fix")
    assert out[3] == ""
    assert out[0] == "hello"


def test_stt_without_edit_returns_empty_edited_text():
    with Capture(STT_RESPONSE):
        out = nodes.ElevenLabsPro_STT().transcribe(api_key="k", audio=AUDIO, model="scribe_v2")
    assert len(out) == 4
    assert out[3] == ""


# --------- Forced alignment ---------

def test_forced_alignment_request_and_outputs():
    payload = {
        "characters": [],
        "words": [{"text": "hi", "start": 0.0, "end": 0.4, "loss": 0.1}],
        "loss": 0.25,
    }
    with Capture(payload) as cap:
        words_json, loss = nodes.ElevenLabsPro_ForcedAlignment().align(
            api_key="k", audio=AUDIO, text="hi")
    assert cap.url.endswith("/v1/forced-alignment")
    assert cap.form == {"text": "hi"}
    assert cap.kwargs["files"]["file"][0] == "input.wav"
    assert json.loads(words_json)[0]["end"] == 0.4
    assert loss == 0.25


def test_forced_alignment_words_feed_subtitle_export():
    words = json.dumps([{"text": "hello", "start": 0.0, "end": 0.5, "loss": 0.1},
                        {"text": "world", "start": 0.6, "end": 1.0, "loss": 0.1}])
    out, = nodes.ElevenLabsPro_SubtitleExport().export(words, format="srt")
    assert "hello world" in out


def test_forced_alignment_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        nodes.ElevenLabsPro_ForcedAlignment().align(api_key="k", audio=AUDIO, text=" ")


# --------- Dialogue ---------

def test_dialogue_defaults_unchanged():
    with Capture() as cap:
        nodes.ElevenLabsPro_Dialogue().generate(api_key="k", text1="hi", voice_id1="v1")
    assert cap.url.endswith("/v1/text-to-dialogue")
    assert cap.body == {
        "inputs": [{"text": "hi", "voice_id": "v1"}],
        "model_id": "eleven_v3",
        "settings": {"stability": 0.5},
        "apply_text_normalization": "off",
    }
    assert cap.kwargs["params"] == {"output_format": "mp3_44100_192"}


def test_dialogue_eleven_v4_and_new_params():
    locators = [{"pronunciation_dictionary_id": "d1", "version_id": "v1"}]
    with Capture() as cap:
        nodes.ElevenLabsPro_Dialogue().generate(
            api_key="k", text1="hi", voice_id1="v1", model="eleven_v4",
            previous_text=" before ", future_text=" after ", use_pvc_as_ivc=True,
            pronunciation_dictionary_locators=json.dumps(locators), output_format="wav_44100")
    assert cap.body["model_id"] == "eleven_v4"
    assert cap.body["previous_text"] == "before"
    assert cap.body["future_text"] == "after"
    assert cap.body["use_pvc_as_ivc"] is True
    assert cap.body["pronunciation_dictionary_locators"] == locators
    assert cap.kwargs["params"] == {"output_format": "wav_44100"}


def test_dialogue_output_formats_cover_all_formats():
    fmt = nodes.ElevenLabsPro_Dialogue.INPUT_TYPES()["optional"]["output_format"]
    assert set(utils.OUTPUT_FORMATS) == set(fmt[0])
    assert {"mp3_44100_192", "opus_48000_192"} <= set(fmt[0])


def test_dialogue_timestamps_request_and_outputs():
    payload = {
        "audio_base64": base64.b64encode(b"mp3").decode(),
        "alignment": {"characters": ["h"], "character_start_times_seconds": [0.0],
                      "character_end_times_seconds": [0.1]},
        "voice_segments": [{"voice_id": "v1", "start_time_seconds": 0.0, "end_time_seconds": 0.1,
                            "character_start_index": 0, "character_end_index": 1}],
    }
    with Capture(payload) as cap:
        audio, ts, segments = nodes.ElevenLabsPro_DialogueTimestamps().generate(
            api_key="k", text1="hi", voice_id1="v1")
    assert cap.url.endswith("/v1/text-to-dialogue/with-timestamps")
    assert audio is FAKE_DECODED
    assert json.loads(ts)["characters"] == ["h"]
    assert json.loads(segments)[0]["voice_id"] == "v1"


def test_dialogue_timestamps_shares_dialogue_inputs():
    assert (nodes.ElevenLabsPro_DialogueTimestamps.INPUT_TYPES()
            == nodes.ElevenLabsPro_Dialogue.INPUT_TYPES())


def test_dialogue_timestamps_missing_audio_returns_silence():
    with Capture({"alignment": {}, "voice_segments": []}):
        audio, ts, segments = nodes.ElevenLabsPro_DialogueTimestamps().generate(
            api_key="k", text1="hi", voice_id1="v1")
    assert audio["waveform"].shape[-1] > 0
    assert json.loads(segments) == []


# --------- Music / composition plan ---------

def test_music_prompt_request_defaults():
    with Capture() as cap:
        nodes.ElevenLabsPro_Music().generate(api_key="k", prompt="synthwave", duration_seconds=12.0)
    assert cap.body == {
        "model_id": "music_v1",
        "store_for_inpainting": False,
        "sign_with_c2pa": False,
        "prompt": "synthwave",
        "music_length_ms": 12000,
        "force_instrumental": False,
    }
    assert cap.kwargs["params"] == {"output_format": "mp3_44100_128"}


def test_music_plan_request_has_no_prompt_only_fields():
    plan = {"positive_global_styles": ["pop"], "negative_global_styles": [], "sections": []}
    with Capture() as cap:
        nodes.ElevenLabsPro_Music().generate(
            api_key="k", prompt="ignored", composition_plan=json.dumps(plan),
            respect_sections_durations=False, lyrics_text="la la", generation_mode="loop",
            use_phonetic_names=True)
    assert cap.body["composition_plan"] == plan
    assert cap.body["respect_sections_durations"] is False
    for field in ("prompt", "music_length_ms", "force_instrumental", "lyrics_text",
                  "generation_mode", "use_phonetic_names"):
        assert field not in cap.body


def test_music_new_models_and_prompt_params():
    with Capture() as cap:
        nodes.ElevenLabsPro_Music().generate(
            api_key="k", prompt="synthwave", model="music_v2_5", output_format="mp3_48000_192",
            finetune_id=" ft_1 ", finetune_strength=1.5, use_phonetic_names=True,
            generation_mode="ambience", lyrics_text=" la la ")
    body = cap.body
    assert body["model_id"] == "music_v2_5"
    assert body["finetune_id"] == "ft_1"
    assert body["finetune_strength"] == 1.5
    assert body["use_phonetic_names"] is True
    assert body["generation_mode"] == "ambience"
    assert body["lyrics_text"] == "la la"
    assert cap.kwargs["params"] == {"output_format": "mp3_48000_192"}


def test_music_finetune_strength_ignored_without_finetune_id():
    with Capture() as cap:
        nodes.ElevenLabsPro_Music().generate(api_key="k", prompt="x", finetune_strength=1.5)
    assert "finetune_id" not in cap.body
    assert "finetune_strength" not in cap.body


def test_music_output_formats_include_48k_mp3():
    fmt = nodes.ElevenLabsPro_Music.INPUT_TYPES()["optional"]["output_format"][0]
    assert {"mp3_48000_128", "mp3_48000_192", "mp3_48000_240", "mp3_48000_320"} <= set(fmt)
    assert set(utils.OUTPUT_FORMATS) <= set(fmt)


def test_music_plan_node_request():
    plan = {"positive_global_styles": ["pop"], "negative_global_styles": [], "sections": []}
    with Capture(plan) as cap:
        out, = nodes.ElevenLabsPro_MusicPlan().plan(
            api_key="k", prompt=" a pop song ", model="music_v2", duration_seconds=45.0)
    assert cap.url.endswith("/v1/music/plan")
    assert cap.body == {"prompt": "a pop song", "model_id": "music_v2", "music_length_ms": 45000}
    assert json.loads(out) == plan


def test_music_plan_node_length_optional_and_validated():
    with Capture({}) as cap:
        nodes.ElevenLabsPro_MusicPlan().plan(api_key="k", prompt="x")
    assert cap.body == {"prompt": "x", "model_id": "music_v1"}
    with pytest.raises(ValueError, match="between 3 and 600"):
        nodes.ElevenLabsPro_MusicPlan().plan(api_key="k", prompt="x", duration_seconds=2.0)
    with pytest.raises(ValueError, match="empty"):
        nodes.ElevenLabsPro_MusicPlan().plan(api_key="k", prompt=" ")
