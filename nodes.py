"""
ElevenLabs nodes for ComfyUI.

15 core API nodes + 11 utility nodes (v2.1):

  Voice:    VoiceSelector, FetchVoices, GetVoiceByName, VoiceClone,
            VoiceDesign, VoiceCreate
  TTS:      TTS, TTSTimestamps, Dialogue, VoiceTagInserter
  Audio:    STS, SFX, AudioIsolation, STT
  Music:    Music
  Utils:    SubtitleExport, AudioConcat, AudioNormalize, AudioMetadata,
            AudioChannels, AudioTrim, CostEstimator, VoiceSettingsPreset,
            SaveAudio
  Config:   APIKey, AccountInfo
"""

import base64
import io
import json
import os
import re
import time

import torch

try:
    from .shared.node_utils import AlwaysExecuteMixin, InputCacheMixin
except ImportError:
    from shared.node_utils import AlwaysExecuteMixin, InputCacheMixin

try:
    from .utils import (
        ELEVENLABS_API_BASE,
        TTS_MODELS,
        TTS_MODELS_REQUIRING_LANGUAGE,
        TTS_MODELS_NORMALIZATION_OFF,
        STS_MODELS,
        STT_MODELS,
        SFX_MODELS,
        SFX_MAX_DURATION,
        MUSIC_MODELS,
        MUSIC_MAX_SECONDS,
        OUTPUT_FORMATS,
        LANGUAGE_OPTIONS,
        LANGUAGE_MAP,
        VOICE_OPTIONS,
        VOICE_MAP,
        V3_AUDIO_TAGS,
        get_api_key,
        api_headers,
        api_post,
        api_get,
        check_response,
        fetch_all_voices,
        validate_text_length,
        audio_bytes_to_comfy,
        comfy_audio_to_bytes,
        silence_audio,
        redact_account_pii,
    )
except ImportError:
    from utils import (
    ELEVENLABS_API_BASE,
    TTS_MODELS,
    TTS_MODELS_REQUIRING_LANGUAGE,
    TTS_MODELS_NORMALIZATION_OFF,
    STS_MODELS,
    STT_MODELS,
    SFX_MODELS,
    SFX_MAX_DURATION,
    MUSIC_MODELS,
    MUSIC_MAX_SECONDS,
    OUTPUT_FORMATS,
    LANGUAGE_OPTIONS,
    LANGUAGE_MAP,
    VOICE_OPTIONS,
    VOICE_MAP,
    V3_AUDIO_TAGS,
    get_api_key,
    api_headers,
    api_post,
    api_get,
    check_response,
    fetch_all_voices,
    validate_text_length,
    audio_bytes_to_comfy,
    comfy_audio_to_bytes,
    silence_audio,
    redact_account_pii,
)

# Shared tooltips reused across generative nodes
_TT_STABILITY = (
    "Voice stability. Lower = more expressive/emotional, "
    "Higher = more consistent/monotone. Creative(<0.5), Natural(0.5), Robust(>0.5)."
)
_TT_SIMILARITY = (
    "How closely AI adheres to the original voice. Too high may introduce artifacts."
)
_TT_STYLE = (
    "Style exaggeration. Increases expressiveness but reduces stability. "
    "Recommended: 0 for most use cases."
)
_TT_SPEED = "Speech speed. 1.0 = normal, <1.0 = slower, >1.0 = faster."
_TT_SPEAKER_BOOST = "Boost similarity to original speaker. Increases latency slightly."
_TT_OUTPUT_FMT = "Audio output format. mp3_44100_192 and opus require Creator tier+."
_TT_SEED = "Seed for reproducibility. 0 = random. Determinism not guaranteed."
_TT_PREVIOUS_TEXT = (
    "Context only — text that came BEFORE this segment. Does NOT generate "
    "audio for this text. Used for seamless chunking across long inputs."
)
_TT_NEXT_TEXT = (
    "Context only — text that comes AFTER this segment. Does NOT generate "
    "audio for this text. Used so the model can anticipate pacing."
)
_TT_ENABLE_LOGGING = (
    "If False, requests zero-retention mode (audio + text not stored by "
    "ElevenLabs). Required for HIPAA / privacy-sensitive content."
)

CATEGORY_BASE = "ElevenLabs Pro"


def _enforce_v3_normalization(model: str, requested: str) -> str:
    """eleven_v3 forbids text normalization 'on'/'auto' per ElevenLabs spec.

    Silently downgrade to 'off' so the user's workflow doesn't 422 on submit.
    """
    if model in TTS_MODELS_NORMALIZATION_OFF and requested != "off":
        print(f"[ElevenLabs Pro] {model} requires apply_text_normalization='off' — overriding.")
        return "off"
    return requested


# ============================================================
# Voice Nodes
# ============================================================

class ElevenLabsPro_VoiceSelector:
    """Select from predefined ElevenLabs voices or enter a custom voice_id."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice": (VOICE_OPTIONS, {"default": VOICE_OPTIONS[1]}),
            },
            "optional": {
                "custom_voice_id": ("STRING", {
                    "default": "",
                    "tooltip": "Override dropdown selection with a custom voice_id. Takes priority if non-empty.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "select"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def select(self, voice, custom_voice_id=""):
        if custom_voice_id and custom_voice_id.strip():
            return (custom_voice_id.strip(),)
        vid = VOICE_MAP.get(voice)
        if not vid:
            raise ValueError(
                f"No voice_id for '{voice}'. Use a custom voice_id or select a named voice."
            )
        return (vid,)


class ElevenLabsPro_FetchVoices(AlwaysExecuteMixin):
    """Fetch all voices from your ElevenLabs account (library + cloned + generated).

    AlwaysExecute is intentional here — fetching is cheap and cache-aware
    inside utils.fetch_all_voices (5-min LRU).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.",
                }),
            },
            "optional": {
                "search": ("STRING", {
                    "default": "",
                    "tooltip": "Filter voices by name, description, accent, or use_case.",
                }),
                "category": (["all", "premade", "cloned", "generated", "professional"], {"default": "all"}),
                "gender": (["any", "male", "female", "nonbinary"], {"default": "any"}),
                "force_refresh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Bypass voice cache and fetch fresh from API.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT",)
    RETURN_NAMES = ("voices_json", "voice_id_list", "count",)
    FUNCTION = "fetch"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def fetch(self, api_key, search="", category="all", gender="any", force_refresh=False):
        key = get_api_key(api_key)
        voices = fetch_all_voices(key, force_refresh=force_refresh)

        if search and search.strip():
            q = search.strip().lower()
            voices = [
                v for v in voices
                if q in v["name"].lower()
                or q in v.get("accent", "").lower()
                or q in v.get("description", "").lower()
                or q in v.get("use_case", "").lower()
            ]

        if category != "all":
            voices = [v for v in voices if v.get("category", "") == category]

        if gender != "any":
            voices = [v for v in voices if v.get("gender", "").lower() == gender]

        voices_json = json.dumps(voices, indent=2)
        id_list = "\n".join(f"{v['voice_id']}  # {v['name']}" for v in voices)
        return (voices_json, id_list, len(voices),)


class ElevenLabsPro_GetVoiceByName(AlwaysExecuteMixin):
    """Look up a single voice_id by (case-insensitive) name. Convenience over FetchVoices."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "name": ("STRING", {"default": "Rachel", "tooltip": "Exact or partial voice name (case-insensitive)."}),
            },
            "optional": {
                "match_mode": (["exact", "contains", "starts_with"], {"default": "contains"}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("voice_id", "matched_name",)
    FUNCTION = "lookup"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def lookup(self, api_key, name, match_mode="contains", force_refresh=False):
        key = get_api_key(api_key)
        voices = fetch_all_voices(key, force_refresh=force_refresh)
        q = (name or "").strip().lower()
        if not q:
            raise ValueError("name input is empty.")
        for v in voices:
            nl = v["name"].lower()
            if match_mode == "exact" and nl == q:
                return (v["voice_id"], v["name"])
            if match_mode == "contains" and q in nl:
                return (v["voice_id"], v["name"])
            if match_mode == "starts_with" and nl.startswith(q):
                return (v["voice_id"], v["name"])
        # Also try predefined voices as a fallback (works without API key call for cached data)
        for vid, vname, _, _ in []:
            pass  # noop; covered by fetch_all_voices already
        raise ValueError(f"No voice matched name='{name}' (match_mode={match_mode}).")


class ElevenLabsPro_VoiceClone(InputCacheMixin):
    """Create an instant voice clone from audio samples (up to 8 files).

    SAFETY: Set `create=False` to prevent accidental clone creation on every
    queue. Each clone burns voice slots and credits. Default is False — flip
    to True only when you actually want to upload. Successful cloning emits
    voice_id; preview/dry-run returns empty string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "voice_name": ("STRING", {"default": "My Cloned Voice", "tooltip": "Name for the cloned voice."}),
                "audio1": ("AUDIO",),
                "create": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "MUST be True to actually create the voice. Default False prevents accidental credit/slot burn on workflow re-queue.",
                }),
            },
            "optional": {
                "audio2": ("AUDIO",),
                "audio3": ("AUDIO",),
                "audio4": ("AUDIO",),
                "audio5": ("AUDIO",),
                "audio6": ("AUDIO",),
                "audio7": ("AUDIO",),
                "audio8": ("AUDIO",),
                "description": ("STRING", {"default": "", "tooltip": "Description of the voice."}),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("voice_id", "status",)
    FUNCTION = "clone"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def clone(self, api_key, voice_name, audio1, create=False,
              audio2=None, audio3=None, audio4=None,
              audio5=None, audio6=None, audio7=None, audio8=None,
              description="", remove_background_noise=False):

        provided_samples = [a for a in [audio1, audio2, audio3, audio4,
                                         audio5, audio6, audio7, audio8] if a is not None]

        if not create:
            return ("", f"DRY-RUN: would clone '{voice_name}' from {len(provided_samples)} samples. Set create=True to upload.")

        key = get_api_key(api_key)

        files = []
        for i, audio in enumerate([audio1, audio2, audio3, audio4,
                                    audio5, audio6, audio7, audio8]):
            if audio is None:
                continue
            audio_bytes, _ = comfy_audio_to_bytes(audio)
            files.append(("files", (f"sample_{i+1}.wav", audio_bytes, "audio/wav")))

        if not files:
            raise ValueError("VoiceClone requires at least one audio sample (audio1).")

        data = {"name": voice_name}
        if description:
            data["description"] = description
        if remove_background_noise:
            data["remove_background_noise"] = "true"

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/voices/add",
            key,
            timeout=120,
            headers={"xi-api-key": key},
            data=data,
            files=files,
        )
        check_response(resp, api_key=key)
        voice_id = resp.json().get("voice_id", "")
        print(f"[ElevenLabs Pro] Voice cloned: {voice_id} ({voice_name})")
        return (voice_id, f"CREATED voice_id={voice_id}")


class ElevenLabsPro_VoiceDesign(InputCacheMixin):
    """Design new voice previews from a text prompt and voice description.

    Returns ALL 3 previews (batched into a single AUDIO tensor and a
    newline-separated list of generated_voice_ids). Use VoiceCreate with
    the chosen ID to save it permanently.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! This is a preview of the designed voice.",
                    "tooltip": "Sample text to generate the preview with.",
                }),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe the voice you want: age, gender, accent, tone, etc.",
                }),
            },
            "optional": {
                "preview_index": ("INT", {
                    "default": 0, "min": 0, "max": 2,
                    "tooltip": "Which of the 3 previews to expose on preview_audio output (0..2).",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING",)
    RETURN_NAMES = ("preview_audio", "generated_voice_id", "all_voice_ids",)
    FUNCTION = "design"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def design(self, api_key, text, voice_description, preview_index=0):
        if not text or not text.strip():
            raise ValueError("Text is required for voice preview.")
        if not voice_description or not voice_description.strip():
            raise ValueError("Voice description is required.")

        key = get_api_key(api_key)

        body = {
            "text": text.strip(),
            "voice_description": voice_description.strip(),
        }

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-voice/create-previews",
            key,
            timeout=120,
            json=body,
            headers=api_headers(key),
        )
        check_response(resp, api_key=key)
        result = resp.json()

        previews = result.get("previews", [])
        if not previews:
            raise RuntimeError("No voice previews returned by API.")

        # Pick the requested preview index, clamped
        idx = max(0, min(preview_index, len(previews) - 1))
        chosen = previews[idx]
        generated_voice_id = chosen.get("generated_voice_id", "")

        audio_b64 = chosen.get("audio_base_64", "")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            # API returns preview audio as mp3_44100_128 by spec
            audio = audio_bytes_to_comfy(audio_bytes, "mp3_44100_128")
        else:
            audio = silence_audio(1.0)

        # Collect all generated_voice_ids so the user can re-pick a different preview
        all_ids = "\n".join(p.get("generated_voice_id", "") for p in previews)

        return (audio, generated_voice_id, all_ids,)


class ElevenLabsPro_VoiceCreate(InputCacheMixin):
    """Save a designed voice preview as a permanent voice in your library.

    SAFETY: Set `create=True` to actually save (consumes a voice slot).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "generated_voice_id": ("STRING", {
                    "default": "",
                    "tooltip": "The generated_voice_id from Voice Design node.",
                }),
                "voice_name": ("STRING", {
                    "default": "My Designed Voice",
                    "tooltip": "Name to save the voice as.",
                }),
                "create": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "MUST be True to actually save. Default False prevents accidental voice-slot burn.",
                }),
            },
            "optional": {
                "voice_description": ("STRING", {
                    "default": "",
                    "tooltip": "Optional description for the saved voice.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("voice_id", "status",)
    FUNCTION = "create"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def create(self, api_key, generated_voice_id, voice_name, create=False, voice_description=""):
        if not generated_voice_id or not generated_voice_id.strip():
            raise ValueError("generated_voice_id is required. Connect from Voice Design node.")

        if not create:
            return ("", f"DRY-RUN: would save '{voice_name}' (generated_voice_id={generated_voice_id[:12]}...). Set create=True to save.")

        key = get_api_key(api_key)

        body = {
            "voice_name": voice_name,
            "generated_voice_id": generated_voice_id.strip(),
        }
        if voice_description and voice_description.strip():
            body["voice_description"] = voice_description.strip()

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-voice/create-voice-from-preview",
            key,
            timeout=60,
            json=body,
            headers=api_headers(key),
        )
        check_response(resp, api_key=key)
        voice_id = resp.json().get("voice_id", "")
        print(f"[ElevenLabs Pro] Voice created from preview: {voice_id} ({voice_name})")
        return (voice_id, f"CREATED voice_id={voice_id}")


# ============================================================
# Text to Speech
# ============================================================

class ElevenLabsPro_TTS(InputCacheMixin):
    """Full-featured Text to Speech with all ElevenLabs API parameters.

    Cached by input hash — identical inputs (text+voice+settings) will not
    re-call the API on workflow re-queue, saving credits.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to convert to speech. Supports v3 audio tags like [whispers], [laughs], [excited].",
                }),
                "voice_id": ("STRING", {
                    "default": "",
                    "tooltip": "Voice ID. Connect from Voice Selector or Fetch Voices node.",
                }),
                "model": (TTS_MODELS, {"default": "eleven_v3"}),
            },
            "optional": {
                "stability": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": _TT_STABILITY,
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": _TT_SIMILARITY,
                }),
                "style": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": _TT_STYLE,
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01,
                    "tooltip": _TT_SPEED,
                }),
                "use_speaker_boost": ("BOOLEAN", {
                    "default": True,
                    "tooltip": _TT_SPEAKER_BOOST,
                }),
                "language": (LANGUAGE_OPTIONS, {
                    "default": "Auto Detect",
                    "tooltip": "Language for the model. Auto Detect lets the model decide. NOTE: turbo/flash v2.5 models require an explicit language.",
                }),
                "apply_text_normalization": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "Text normalization mode. eleven_v3 is auto-forced to 'off' (the only valid value for that model).",
                }),
                "output_format": (OUTPUT_FORMATS, {
                    "default": "mp3_44100_128",
                    "tooltip": _TT_OUTPUT_FMT,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": _TT_SEED,
                }),
                "previous_text": ("STRING", {
                    "default": "",
                    "tooltip": _TT_PREVIOUS_TEXT,
                }),
                "next_text": ("STRING", {
                    "default": "",
                    "tooltip": _TT_NEXT_TEXT,
                }),
                "use_pvc_as_ivc": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the IVC version of a Professional Voice Clone.",
                }),
                "enable_logging": ("BOOLEAN", {
                    "default": True,
                    "tooltip": _TT_ENABLE_LOGGING,
                }),
                "pronunciation_dictionary_locators": ("STRING", {
                    "default": "",
                    "tooltip": "JSON array of {\"pronunciation_dictionary_id\": ..., \"version_id\": ...} objects.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "raw_response_info",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY_BASE}/TTS"

    def generate(self, api_key, text, voice_id, model,
                 stability=0.5, similarity_boost=0.75, style=0.0, speed=1.0,
                 use_speaker_boost=True, language="Auto Detect",
                 apply_text_normalization="auto", output_format="mp3_44100_128",
                 seed=0, previous_text="", next_text="",
                 use_pvc_as_ivc=False, enable_logging=True,
                 pronunciation_dictionary_locators=""):

        if not text or not text.strip():
            raise ValueError("Text input is empty.")
        if not voice_id or not voice_id.strip():
            raise ValueError("voice_id is required. Connect a Voice Selector node.")

        validate_text_length(text, model)

        key = get_api_key(api_key)
        lang_code = LANGUAGE_MAP.get(language, "")

        # Hard validation: turbo/flash v2.5 require explicit language
        if model in TTS_MODELS_REQUIRING_LANGUAGE and not lang_code:
            raise ValueError(
                f"Model {model} requires an explicit language code. "
                f"Select one from the 'language' dropdown (not 'Auto Detect')."
            )

        body = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "speed": speed,
                "use_speaker_boost": use_speaker_boost,
            },
            "apply_text_normalization": _enforce_v3_normalization(model, apply_text_normalization),
            "use_pvc_as_ivc": bool(use_pvc_as_ivc),
        }

        if lang_code:
            body["language_code"] = lang_code
        if seed > 0:
            body["seed"] = seed
        if previous_text and previous_text.strip():
            body["previous_text"] = previous_text.strip()
        if next_text and next_text.strip():
            body["next_text"] = next_text.strip()

        if pronunciation_dictionary_locators and pronunciation_dictionary_locators.strip():
            try:
                locators = json.loads(pronunciation_dictionary_locators.strip())
                if isinstance(locators, list) and locators:
                    body["pronunciation_dictionary_locators"] = locators
            except json.JSONDecodeError as exc:
                # Surface to user — silent swallow is bad UX and was a regression risk
                raise ValueError(
                    f"pronunciation_dictionary_locators is not valid JSON: {exc}"
                )

        params = {"output_format": output_format}
        if not enable_logging:
            params["enable_logging"] = "false"

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-speech/{voice_id.strip()}",
            key,
            timeout=300,
            json=body,
            params=params,
            headers=api_headers(key),
        )
        check_response(resp, api_key=key)

        audio_bytes = resp.content
        info = json.dumps({
            "model": model,
            "voice_id": voice_id.strip(),
            "format": output_format,
            "bytes": len(audio_bytes),
            "characters": len(text),
            "language_code": lang_code or "auto",
        })

        audio = audio_bytes_to_comfy(audio_bytes, output_format)
        return (audio, info,)


class ElevenLabsPro_TTSTimestamps(InputCacheMixin):
    """Text to Speech with word-level / character-level timestamps as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to convert to speech with timing information.",
                }),
                "voice_id": ("STRING", {
                    "default": "",
                    "tooltip": "Voice ID. Connect from Voice Selector or Fetch Voices node.",
                }),
                "model": (TTS_MODELS, {"default": "eleven_v3"}),
            },
            "optional": {
                "stability": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": _TT_STABILITY,
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": _TT_SIMILARITY,
                }),
                "output_format": (OUTPUT_FORMATS, {
                    "default": "mp3_44100_128",
                    "tooltip": _TT_OUTPUT_FMT,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": _TT_SEED,
                }),
                "enable_logging": ("BOOLEAN", {"default": True, "tooltip": _TT_ENABLE_LOGGING}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "timestamps_json",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY_BASE}/TTS"

    def generate(self, api_key, text, voice_id, model,
                 stability=0.5, similarity_boost=0.75,
                 output_format="mp3_44100_128", seed=0, enable_logging=True):

        if not text or not text.strip():
            raise ValueError("Text input is empty.")
        if not voice_id or not voice_id.strip():
            raise ValueError("voice_id is required.")

        validate_text_length(text, model)

        key = get_api_key(api_key)

        body = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
            "apply_text_normalization": _enforce_v3_normalization(model, "auto"),
        }
        if seed > 0:
            body["seed"] = seed

        params = {"output_format": output_format}
        if not enable_logging:
            params["enable_logging"] = "false"

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-speech/{voice_id.strip()}/with-timestamps",
            key,
            timeout=300,
            json=body,
            params=params,
            headers=api_headers(key),
        )
        check_response(resp, api_key=key)

        result = resp.json()

        # Audio is base64-encoded in the response
        audio_b64 = result.get("audio_base_64", "")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            audio = audio_bytes_to_comfy(audio_bytes, output_format)
        else:
            audio = silence_audio(1.0)

        alignment = result.get("alignment", {})
        timestamps_json = json.dumps(alignment, indent=2)

        return (audio, timestamps_json,)


# ============================================================
# Speech to Speech
# ============================================================

class ElevenLabsPro_STS(InputCacheMixin):
    """Speech to Speech -- transform voice while preserving content and emotion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "audio": ("AUDIO",),
                "voice_id": ("STRING", {"default": "", "tooltip": "Target voice ID."}),
                "model": (STS_MODELS, {"default": "eleven_multilingual_sts_v2"}),
            },
            "optional": {
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TT_STABILITY}),
                "similarity_boost": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TT_SIMILARITY}),
                "style": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TT_STYLE}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": _TT_SPEED}),
                "use_speaker_boost": ("BOOLEAN", {"default": True, "tooltip": _TT_SPEAKER_BOOST}),
                "output_format": (OUTPUT_FORMATS, {"default": "mp3_44100_128", "tooltip": _TT_OUTPUT_FMT}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295, "tooltip": _TT_SEED}),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
                "enable_logging": ("BOOLEAN", {"default": True, "tooltip": _TT_ENABLE_LOGGING}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "transform"
    CATEGORY = f"{CATEGORY_BASE}/Audio"

    def transform(self, api_key, audio, voice_id, model,
                  stability=0.5, similarity_boost=0.75, style=0.0, speed=1.0,
                  use_speaker_boost=True, output_format="mp3_44100_128",
                  seed=0, remove_background_noise=False, enable_logging=True):

        if not voice_id or not voice_id.strip():
            raise ValueError("voice_id is required.")
        key = get_api_key(api_key)
        audio_bytes, _ = comfy_audio_to_bytes(audio)

        voice_settings = json.dumps({
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "speed": speed,
            "use_speaker_boost": use_speaker_boost,
        })

        data = {
            "model_id": model,
            "voice_settings": voice_settings,
        }
        if remove_background_noise:
            data["remove_background_noise"] = "true"
        if seed > 0:
            data["seed"] = str(seed)

        params = {"output_format": output_format}
        if not enable_logging:
            params["enable_logging"] = "false"

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/speech-to-speech/{voice_id.strip()}",
            key,
            timeout=300,
            headers={"xi-api-key": key},
            data=data,
            files={"audio": ("input.wav", audio_bytes, "audio/wav")},
            params=params,
        )
        check_response(resp, api_key=key)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Sound Effects
# ============================================================

class ElevenLabsPro_SFX(InputCacheMixin):
    """Generate sound effects from text descriptions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the sound effect to generate.",
                }),
            },
            "optional": {
                "model": (SFX_MODELS, {
                    "default": "eleven_text_to_sound_v2",
                    "tooltip": "Sound effects model. Default is the current v2 endpoint.",
                }),
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": SFX_MAX_DURATION, "step": 0.1,
                    "tooltip": f"Duration in seconds (0.5-{SFX_MAX_DURATION}).",
                }),
                "prompt_influence": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How closely generation follows the prompt.",
                }),
                "loop": ("BOOLEAN", {"default": False, "tooltip": "Create a smoothly looping sound effect (v2 model only)."}),
                "output_format": (OUTPUT_FORMATS, {
                    "default": "mp3_44100_128",
                    "tooltip": _TT_OUTPUT_FMT,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": _TT_SEED,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY_BASE}/Audio"

    def generate(self, api_key, text, model="eleven_text_to_sound_v2",
                 duration=5.0, prompt_influence=0.3,
                 loop=False, output_format="mp3_44100_128", seed=0):

        if not text or not text.strip():
            raise ValueError("Text description is empty.")
        # Clamp defensively even though slider min/max should prevent OOB
        if duration < 0.5 or duration > SFX_MAX_DURATION:
            raise ValueError(f"duration must be between 0.5 and {SFX_MAX_DURATION} seconds (got {duration}).")

        key = get_api_key(api_key)

        body = {
            "text": text,
            "model_id": model,
            "duration_seconds": duration,
            "prompt_influence": prompt_influence,
        }
        if loop:
            body["loop"] = True
        if seed > 0:
            body["seed"] = seed

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/sound-generation",
            key,
            timeout=300,
            json=body,
            params={"output_format": output_format},
            headers=api_headers(key),
        )
        check_response(resp, api_key=key)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Audio Isolation
# ============================================================

class ElevenLabsPro_AudioIsolation(InputCacheMixin):
    """Remove background noise from audio, isolating vocals/speech."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "audio": ("AUDIO",),
            },
            "optional": {
                "output_format": (OUTPUT_FORMATS, {
                    "default": "mp3_44100_128",
                    "tooltip": _TT_OUTPUT_FMT,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "isolate"
    CATEGORY = f"{CATEGORY_BASE}/Audio"

    def isolate(self, api_key, audio, output_format="mp3_44100_128"):
        key = get_api_key(api_key)
        audio_bytes, _ = comfy_audio_to_bytes(audio)

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/audio-isolation",
            key,
            timeout=300,
            headers={"xi-api-key": key},
            files={"audio": ("input.wav", audio_bytes, "audio/wav")},
            params={"output_format": output_format},
        )
        check_response(resp, api_key=key)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Speech to Text
# ============================================================

class ElevenLabsPro_STT(InputCacheMixin):
    """Transcribe audio to text with speaker diarization and audio event tagging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "audio": ("AUDIO",),
                "model": (STT_MODELS, {"default": "scribe_v2"}),
            },
            "optional": {
                "language_code": ("STRING", {
                    "default": "",
                    "tooltip": "ISO language code (e.g. 'en', 'es'). Empty = auto-detect.",
                }),
                "tag_audio_events": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Annotate sounds like (laughter), (music) in transcript.",
                }),
                "diarize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Identify and label different speakers.",
                }),
                "num_speakers": ("INT", {
                    "default": 0, "min": 0, "max": 32,
                    "tooltip": "Expected number of speakers. 0 = auto-detect.",
                }),
                "timestamps_granularity": (["word", "character", "none"], {
                    "default": "word",
                    "tooltip": "Timestamp granularity. 'none' returns no timing info (matches API spec).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": _TT_SEED,
                }),
                "diarization_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Threshold for speaker diarization. Higher = stricter separation.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sampling temperature. 0 = deterministic.",
                }),
                "keyterms": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated key terms to bias recognition toward.",
                }),
                "no_verbatim": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove filler words and stutters (scribe_v2 only).",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text", "language_code", "words_json",)
    FUNCTION = "transcribe"
    CATEGORY = f"{CATEGORY_BASE}/Audio"

    def transcribe(self, api_key, audio, model,
                   language_code="", tag_audio_events=False, diarize=False,
                   num_speakers=0, timestamps_granularity="word", seed=0,
                   diarization_threshold=0.5, temperature=0.0,
                   keyterms="", no_verbatim=False):

        key = get_api_key(api_key)
        audio_bytes, _ = comfy_audio_to_bytes(audio)

        data = {"model_id": model, "timestamps_granularity": timestamps_granularity}
        if language_code and language_code.strip():
            data["language_code"] = language_code.strip()
        if tag_audio_events:
            data["tag_audio_events"] = "true"
        if diarize:
            data["diarize"] = "true"
        if num_speakers > 0:
            data["num_speakers"] = str(num_speakers)
        if seed > 0:
            data["seed"] = str(seed)
        if diarization_threshold != 0.5:
            data["diarization_threshold"] = str(diarization_threshold)
        if temperature > 0.0:
            data["temperature"] = str(temperature)
        if keyterms and keyterms.strip():
            data["keyterms"] = keyterms.strip()
        if no_verbatim:
            data["no_verbatim"] = "true"

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/speech-to-text",
            key,
            timeout=300,
            headers={"xi-api-key": key},
            data=data,
            files={"file": ("input.wav", audio_bytes, "audio/wav")},
        )
        check_response(resp, api_key=key)

        result = resp.json()
        text = result.get("text", "")
        lang = result.get("language_code", "")
        words = json.dumps(result.get("words", []), indent=2)
        return (text, lang, words,)


# ============================================================
# Dialogue (multi-speaker, v3)
# ============================================================

class ElevenLabsPro_Dialogue(InputCacheMixin):
    """Generate multi-speaker dialogue. Each speaker gets their own voice and text (up to 10)."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "text1": ("STRING", {"multiline": True, "default": "", "tooltip": "Speaker 1 text."}),
                "voice_id1": ("STRING", {"default": "", "tooltip": "Speaker 1 voice ID."}),
                "model": (["eleven_v3"], {"default": "eleven_v3"}),
            },
            "optional": {},
        }

        for i in range(2, 11):
            inputs["optional"][f"text{i}"] = ("STRING", {"multiline": True, "default": ""})
            inputs["optional"][f"voice_id{i}"] = ("STRING", {"default": ""})

        inputs["optional"].update({
            "stability": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": _TT_STABILITY,
            }),
            "apply_text_normalization": (["auto", "on", "off"], {"default": "off",
                "tooltip": "eleven_v3 requires 'off' — keeping default."}),
            "language": (LANGUAGE_OPTIONS, {"default": "Auto Detect"}),
            "output_format": (["mp3_44100_192", "opus_48000_192"], {
                "default": "mp3_44100_192",
                "tooltip": _TT_OUTPUT_FMT,
            }),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 4294967295,
                "tooltip": _TT_SEED,
            }),
            "enable_logging": ("BOOLEAN", {"default": True, "tooltip": _TT_ENABLE_LOGGING}),
        })
        return inputs

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY_BASE}/TTS"

    def generate(self, api_key, text1, voice_id1, model="eleven_v3", **kwargs):
        key = get_api_key(api_key)
        lang_code = LANGUAGE_MAP.get(kwargs.get("language", "Auto Detect"), "")

        inputs = []
        all_pairs = [(text1, voice_id1)]
        for i in range(2, 11):
            t = kwargs.get(f"text{i}", "")
            v = kwargs.get(f"voice_id{i}", "")
            all_pairs.append((t, v))

        total_chars = 0
        for text, vid in all_pairs:
            if text and text.strip() and vid and vid.strip():
                inputs.append({"text": text.strip(), "voice_id": vid.strip()})
                total_chars += len(text.strip())

        if not inputs:
            raise ValueError("At least one text + voice_id pair is required.")

        # Validate total text length against the model limit. The dialogue endpoint
        # consumes the concatenation, so use the same per-model limit table.
        validate_text_length("x" * total_chars, model)

        stability = kwargs.get("stability", 0.5)
        apply_text_normalization = _enforce_v3_normalization(
            model, kwargs.get("apply_text_normalization", "off"),
        )
        output_format = kwargs.get("output_format", "mp3_44100_192")
        seed = kwargs.get("seed", 0)
        enable_logging = kwargs.get("enable_logging", True)

        body = {
            "inputs": inputs,
            "model_id": model,
            "settings": {"stability": stability},
            "apply_text_normalization": apply_text_normalization,
        }
        if lang_code:
            body["language_code"] = lang_code
        if seed > 0:
            body["seed"] = seed

        params = {"output_format": output_format}
        if not enable_logging:
            params["enable_logging"] = "false"

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-dialogue",
            key,
            timeout=300,
            json=body,
            params=params,
            headers=api_headers(key),
        )
        check_response(resp, api_key=key)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Music Generation
# ============================================================

class ElevenLabsPro_Music(InputCacheMixin):
    """Generate music from a text prompt OR a composition plan.

    Updated to match current ElevenLabs API: uses `music_length_ms`
    (3000-600000ms), supports optional `composition_plan` JSON to override
    the simple `prompt`, `respect_sections_durations`, `store_for_inpainting`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text description of the music. Required unless composition_plan is set.",
                }),
            },
            "optional": {
                "model": (MUSIC_MODELS, {"default": "music_v1"}),
                "duration_seconds": ("FLOAT", {
                    "default": 30.0, "min": 3.0, "max": MUSIC_MAX_SECONDS, "step": 1.0,
                    "tooltip": f"Duration in seconds (3 - {int(MUSIC_MAX_SECONDS)}). Converted to music_length_ms.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": _TT_SEED,
                }),
                "force_instrumental": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate instrumental-only music (no vocals).",
                }),
                "respect_sections_durations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enforce composition_plan section durations.",
                }),
                "store_for_inpainting": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Server-side store result for follow-up inpainting calls.",
                }),
                "sign_with_c2pa": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Sign output with C2PA provenance.",
                }),
                "composition_plan": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional JSON composition_plan object — overrides prompt if set. See ElevenLabs Music API docs.",
                }),
                "output_format": (OUTPUT_FORMATS, {
                    "default": "mp3_44100_128",
                    "tooltip": _TT_OUTPUT_FMT,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY_BASE}/Music"

    def generate(self, api_key, prompt, model="music_v1",
                 duration_seconds=30.0, seed=0,
                 force_instrumental=False,
                 respect_sections_durations=True,
                 store_for_inpainting=False,
                 sign_with_c2pa=False,
                 composition_plan="",
                 output_format="mp3_44100_128"):

        has_plan = bool(composition_plan and composition_plan.strip())
        if not has_plan and (not prompt or not prompt.strip()):
            raise ValueError("Music requires either a prompt OR a composition_plan.")

        if not (3.0 <= duration_seconds <= MUSIC_MAX_SECONDS):
            raise ValueError(
                f"duration_seconds must be between 3 and {int(MUSIC_MAX_SECONDS)} (got {duration_seconds})."
            )

        key = get_api_key(api_key)

        body = {
            "model_id": model,
            "music_length_ms": int(round(duration_seconds * 1000)),
            "force_instrumental": bool(force_instrumental),
            "respect_sections_durations": bool(respect_sections_durations),
            "store_for_inpainting": bool(store_for_inpainting),
            "sign_with_c2pa": bool(sign_with_c2pa),
        }
        if has_plan:
            try:
                body["composition_plan"] = json.loads(composition_plan.strip())
            except json.JSONDecodeError as exc:
                raise ValueError(f"composition_plan is not valid JSON: {exc}")
        else:
            body["prompt"] = prompt.strip()
        if seed > 0:
            body["seed"] = seed

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/music",
            key,
            timeout=600,
            json=body,
            params={"output_format": output_format},
            headers=api_headers(key),
        )
        check_response(resp, api_key=key)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Config / Account
# ============================================================

class ElevenLabsPro_APIKey:
    """Store your ElevenLabs API key. Connect to other ElevenLabs nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "Your ElevenLabs API key. Also checks ELEVENLABS_API_KEY env var as fallback.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "passthrough"
    CATEGORY = f"{CATEGORY_BASE}/Config"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Hash by key value (not by literal value to avoid leaking into PNG metadata)
        import hashlib
        k = kwargs.get("api_key", "") or ""
        return hashlib.sha256(k.encode()).hexdigest()

    def passthrough(self, api_key):
        key = get_api_key(api_key)
        return (key,)


class ElevenLabsPro_AccountInfo(AlwaysExecuteMixin):
    """Retrieve ElevenLabs account info: subscription tier, character usage.

    PII fields (email, name, billing, IDs) are REDACTED before being returned
    so they cannot leak into a saved workflow PNG / shared on Civitai.
    Uses /v1/user/subscription for cleaner extended subscription data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
            },
            "optional": {
                "endpoint": (["user_subscription", "user"], {
                    "default": "user_subscription",
                    "tooltip": "Which endpoint to query. user_subscription is preferred (no PII).",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT",)
    RETURN_NAMES = ("account_json_redacted", "subscription_tier", "character_count", "character_limit",)
    FUNCTION = "fetch"
    CATEGORY = f"{CATEGORY_BASE}/Config"

    def fetch(self, api_key, endpoint="user_subscription"):
        key = get_api_key(api_key)

        url = (
            f"{ELEVENLABS_API_BASE}/v1/user/subscription"
            if endpoint == "user_subscription"
            else f"{ELEVENLABS_API_BASE}/v1/user"
        )
        resp = api_get(url, key)
        check_response(resp, api_key=key)

        data = resp.json()
        # /v1/user/subscription returns the subscription dict directly;
        # /v1/user wraps it under "subscription".
        sub = data if endpoint == "user_subscription" else data.get("subscription", {})
        tier = sub.get("tier", "unknown")
        char_count = sub.get("character_count", 0)
        char_limit = sub.get("character_limit", 0)

        redacted = redact_account_pii(data)
        account_json = json.dumps(redacted, indent=2)
        return (account_json, tier, char_count, char_limit,)


# ============================================================
# Utility nodes (new in v2.1) — pure-Python, no API cost
# ============================================================

class ElevenLabsPro_VoiceTagInserter:
    """Inject v3 expression tags like [whispers], [laughs] into a text prompt.

    Pure-text helper — no API call. Builds `before_tag text after_tag` style.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "tag": (V3_AUDIO_TAGS, {"default": "whispers"}),
            },
            "optional": {
                "placement": (["prepend", "append", "wrap"], {"default": "prepend"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_with_tag",)
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY_BASE}/Utils/Text"

    def build(self, text, tag, placement="prepend"):
        marker = f"[{tag}]"
        t = (text or "").strip()
        if not t:
            return (marker,)
        if placement == "prepend":
            return (f"{marker} {t}",)
        if placement == "append":
            return (f"{t} {marker}",)
        return (f"{marker} {t} {marker}",)


class ElevenLabsPro_SubtitleExport:
    """Export TTSTimestamps `timestamps_json` (or STT `words_json`) to SRT or VTT.

    Pure-Python, no API call. Groups characters/words into cues by length / time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timestamps_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Output of TTSTimestamps or STT (words list).",
                }),
                "format": (["srt", "vtt"], {"default": "srt"}),
            },
            "optional": {
                "max_chars_per_cue": ("INT", {"default": 42, "min": 10, "max": 200}),
                "max_seconds_per_cue": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subtitle_text",)
    FUNCTION = "export"
    CATEGORY = f"{CATEGORY_BASE}/Utils/Text"

    def export(self, timestamps_json, format="srt",
               max_chars_per_cue=42, max_seconds_per_cue=6.0):
        try:
            data = json.loads(timestamps_json) if timestamps_json.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"timestamps_json is not valid JSON: {e}")

        # Two formats: TTSTimestamps alignment dict OR STT words list
        cues = []
        if isinstance(data, dict) and "characters" in data:
            # TTSTimestamps alignment: characters[], character_start_times_seconds[], character_end_times_seconds[]
            chars = data.get("characters") or []
            starts = data.get("character_start_times_seconds") or []
            ends = data.get("character_end_times_seconds") or []
            n = min(len(chars), len(starts), len(ends))
            if n == 0:
                return ("",)
            cur = {"text": "", "start": starts[0], "end": ends[0]}
            for i in range(n):
                ch = chars[i]
                proposed = cur["text"] + ch
                proposed_len = len(proposed)
                duration = ends[i] - cur["start"]
                if (proposed_len > max_chars_per_cue or duration > max_seconds_per_cue) and cur["text"].strip():
                    cues.append(cur)
                    cur = {"text": ch, "start": starts[i], "end": ends[i]}
                else:
                    cur["text"] = proposed
                    cur["end"] = ends[i]
            if cur["text"].strip():
                cues.append(cur)
        elif isinstance(data, list):
            # STT word list: each item { text, start, end, type }
            cur = None
            for w in data:
                if not isinstance(w, dict):
                    continue
                wt = w.get("text", "")
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", ws))
                if cur is None:
                    cur = {"text": wt, "start": ws, "end": we}
                    continue
                proposed = (cur["text"] + " " + wt).strip()
                if (len(proposed) > max_chars_per_cue
                        or (we - cur["start"]) > max_seconds_per_cue):
                    cues.append(cur)
                    cur = {"text": wt, "start": ws, "end": we}
                else:
                    cur["text"] = proposed
                    cur["end"] = we
            if cur:
                cues.append(cur)
        else:
            raise ValueError("Unsupported timestamps_json shape. Expected TTSTimestamps alignment or STT words list.")

        if format == "srt":
            out_lines = []
            for i, c in enumerate(cues, start=1):
                out_lines.append(str(i))
                out_lines.append(f"{_fmt_srt_time(c['start'])} --> {_fmt_srt_time(c['end'])}")
                out_lines.append(c["text"].strip())
                out_lines.append("")
            return ("\n".join(out_lines),)
        else:  # vtt
            out_lines = ["WEBVTT", ""]
            for c in cues:
                out_lines.append(f"{_fmt_vtt_time(c['start'])} --> {_fmt_vtt_time(c['end'])}")
                out_lines.append(c["text"].strip())
                out_lines.append("")
            return ("\n".join(out_lines),)


def _fmt_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _fmt_vtt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


class ElevenLabsPro_AudioConcat:
    """Concatenate up to 8 AUDIO inputs along the time axis.

    Resamples to the first non-None input's sample rate if mismatched.
    Mixed channel counts: mono is broadcast to stereo to match.
    Pure-Python, no API call.
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(1, 9):
            opt[f"audio{i}"] = ("AUDIO",)
        opt["gap_seconds"] = ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.05,
            "tooltip": "Silence inserted between each clip.",
        })
        return {"required": {}, "optional": opt}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concat"
    CATEGORY = f"{CATEGORY_BASE}/Utils/Audio"

    def concat(self, **kwargs):
        clips = []
        for i in range(1, 9):
            a = kwargs.get(f"audio{i}")
            if a is not None:
                clips.append(a)
        if not clips:
            raise ValueError("AudioConcat requires at least one audio input.")
        target_sr = int(clips[0]["sample_rate"])
        gap = float(kwargs.get("gap_seconds", 0.0))

        # Find max channels across clips so we can broadcast mono -> N
        max_ch = max(_clip_channels(c) for c in clips)

        out_pieces = []
        for i, c in enumerate(clips):
            wf = c["waveform"]
            sr = int(c["sample_rate"])
            # Normalize shape to [1, C, T]
            if wf.dim() == 1: wf = wf.unsqueeze(0).unsqueeze(0)
            elif wf.dim() == 2: wf = wf.unsqueeze(0)
            # Resample if needed
            if sr != target_sr:
                wf = _resample(wf, sr, target_sr)
            # Broadcast channels
            if wf.shape[1] != max_ch:
                if wf.shape[1] == 1 and max_ch > 1:
                    wf = wf.expand(-1, max_ch, -1).contiguous()
                else:
                    # Mix to mono then broadcast
                    mono = wf.mean(dim=1, keepdim=True)
                    wf = mono.expand(-1, max_ch, -1).contiguous()
            out_pieces.append(wf)
            if i < len(clips) - 1 and gap > 0:
                gap_samples = int(round(gap * target_sr))
                out_pieces.append(torch.zeros(1, max_ch, gap_samples, dtype=wf.dtype))

        out = torch.cat(out_pieces, dim=2)
        return ({"waveform": out, "sample_rate": target_sr},)


def _clip_channels(a: dict) -> int:
    wf = a["waveform"]
    if wf.dim() == 1: return 1
    if wf.dim() == 2: return wf.shape[0]
    if wf.dim() == 3: return wf.shape[1]
    return 1


def _resample(wf: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    """Resample [1, C, T] tensor from src_sr to dst_sr.

    Tries torchaudio first; falls back to torch interpolation (lower quality).
    """
    if src_sr == dst_sr:
        return wf
    try:
        import torchaudio
        # torchaudio expects [C, T] or [B, C, T]; functional.resample handles both
        return torchaudio.functional.resample(wf, src_sr, dst_sr)
    except Exception:
        # Linear interp fallback
        B, C, T = wf.shape
        new_T = int(round(T * dst_sr / src_sr))
        # interpolate operates on [B, C, T]
        return torch.nn.functional.interpolate(wf, size=new_T, mode="linear", align_corners=False)


class ElevenLabsPro_AudioNormalize:
    """Normalize AUDIO to peak or RMS target. Pure-Python, no API call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {
                "mode": (["peak", "rms"], {"default": "peak"}),
                "target_db": ("FLOAT", {
                    "default": -1.0, "min": -60.0, "max": 0.0, "step": 0.1,
                    "tooltip": "Target dBFS. -1 = nearly full-scale; -16 ≈ broadcast loudness.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "normalize"
    CATEGORY = f"{CATEGORY_BASE}/Utils/Audio"

    def normalize(self, audio, mode="peak", target_db=-1.0):
        wf = audio["waveform"].float()
        if wf.numel() == 0:
            return (audio,)
        target_amp = 10.0 ** (target_db / 20.0)
        if mode == "peak":
            current = float(wf.abs().max().item())
        else:
            current = float(torch.sqrt((wf ** 2).mean()).item())
        if current < 1e-9:
            return (audio,)
        gain = target_amp / current
        return ({"waveform": wf * gain, "sample_rate": int(audio["sample_rate"])},)


class ElevenLabsPro_AudioMetadata:
    """Read duration, channels, sample rate from an AUDIO dict."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT",)
    RETURN_NAMES = ("duration_seconds", "sample_rate", "channels", "samples",)
    FUNCTION = "inspect"
    CATEGORY = f"{CATEGORY_BASE}/Utils/Audio"

    def inspect(self, audio):
        wf = audio["waveform"]
        sr = int(audio["sample_rate"])
        if wf.dim() == 3:
            _, c, t = wf.shape
        elif wf.dim() == 2:
            c, t = wf.shape
        elif wf.dim() == 1:
            c, t = 1, wf.shape[0]
        else:
            c, t = 1, 0
        dur = t / sr if sr > 0 else 0.0
        return (float(dur), sr, int(c), int(t),)


class ElevenLabsPro_AudioChannels:
    """Convert AUDIO between mono / stereo or extract single channel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["to_mono", "to_stereo", "left_only", "right_only", "swap_channels"], {
                    "default": "to_mono",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    CATEGORY = f"{CATEGORY_BASE}/Utils/Audio"

    def convert(self, audio, mode="to_mono"):
        wf = audio["waveform"]
        if wf.dim() == 2: wf = wf.unsqueeze(0)
        if wf.dim() == 1: wf = wf.unsqueeze(0).unsqueeze(0)
        B, C, T = wf.shape
        if mode == "to_mono":
            out = wf.mean(dim=1, keepdim=True)
        elif mode == "to_stereo":
            if C == 1:
                out = wf.expand(-1, 2, -1).contiguous()
            elif C >= 2:
                out = wf[:, :2, :]
            else:
                out = wf
        elif mode == "left_only":
            out = wf[:, :1, :] if C >= 1 else wf
        elif mode == "right_only":
            out = wf[:, 1:2, :] if C >= 2 else wf[:, :1, :]
        elif mode == "swap_channels":
            if C >= 2:
                out = wf.clone()
                out[:, 0, :], out[:, 1, :] = wf[:, 1, :], wf[:, 0, :]
            else:
                out = wf
        else:
            out = wf
        return ({"waveform": out, "sample_rate": int(audio["sample_rate"])},)


class ElevenLabsPro_AudioTrim:
    """Trim AUDIO by start and end seconds. Pure-Python, no API call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
                "end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01,
                                          "tooltip": "0 = trim to end of clip."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim"
    CATEGORY = f"{CATEGORY_BASE}/Utils/Audio"

    def trim(self, audio, start_seconds=0.0, end_seconds=0.0):
        wf = audio["waveform"]
        sr = int(audio["sample_rate"])
        # Normalize to [B, C, T]
        if wf.dim() == 2: wf = wf.unsqueeze(0)
        if wf.dim() == 1: wf = wf.unsqueeze(0).unsqueeze(0)
        T = wf.shape[-1]
        start = max(0, int(round(start_seconds * sr)))
        end = T if end_seconds <= 0 else min(T, int(round(end_seconds * sr)))
        if start >= end:
            return (silence_audio(0.01, sample_rate=sr, channels=wf.shape[1]),)
        return ({"waveform": wf[:, :, start:end].contiguous(), "sample_rate": sr},)


class ElevenLabsPro_CostEstimator:
    """Estimate ElevenLabs credit cost BEFORE making the API call.

    Pure-Python, no API call. Uses standard ElevenLabs character-billing rates.
    Returns conservative upper-bound estimates.
    """

    # Approximate credit cost per character for each TTS model (1 credit = 1 char for v2/turbo,
    # 3x for flash, etc — these are conservative; refer to ElevenLabs pricing).
    _COST_PER_CHAR = {
        "eleven_v3": 1.0,
        "eleven_multilingual_v2": 1.0,
        "eleven_flash_v2_5": 0.5,
        "eleven_turbo_v2_5": 0.5,
        "eleven_flash_v2": 0.5,
        "eleven_turbo_v2": 0.5,
        "eleven_multilingual_v1": 1.0,
        "eleven_monolingual_v1": 1.0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model": (TTS_MODELS, {"default": "eleven_v3"}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("characters", "estimated_credits", "summary",)
    FUNCTION = "estimate"
    CATEGORY = f"{CATEGORY_BASE}/Utils"

    def estimate(self, text, model="eleven_v3"):
        chars = len(text or "")
        per = self._COST_PER_CHAR.get(model, 1.0)
        cost = chars * per
        summary = f"{chars} chars × {per:.2f} = ~{cost:.0f} credits ({model})"
        return (chars, float(cost), summary,)


class ElevenLabsPro_VoiceSettingsPreset:
    """Build / serialize a voice settings preset as JSON.

    Connect the output to any node that accepts a `voice_settings` JSON
    string (or use it as documentation). Pure-Python, no API call.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TT_STABILITY}),
                "similarity_boost": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TT_SIMILARITY}),
                "style": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TT_STYLE}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": _TT_SPEED}),
                "use_speaker_boost": ("BOOLEAN", {"default": True, "tooltip": _TT_SPEAKER_BOOST}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_settings_json",)
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY_BASE}/Utils"

    def build(self, stability, similarity_boost, style, speed, use_speaker_boost):
        return (json.dumps({
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "speed": speed,
            "use_speaker_boost": use_speaker_boost,
        }, indent=2),)


class ElevenLabsPro_SaveAudio:
    """Save AUDIO to disk under ComfyUI's output directory.

    Sanitizes the filename (no .. or path separators) so a malicious
    workflow can't write outside the output directory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "elevenlabs"}),
                "format": (["wav", "mp3", "flac", "ogg"], {"default": "wav"}),
            },
            "optional": {
                "subdirectory": ("STRING", {"default": "", "tooltip": "Optional subdir under ComfyUI/output."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = f"{CATEGORY_BASE}/Utils"

    def save(self, audio, filename_prefix="elevenlabs", format="wav", subdirectory=""):
        try:
            import folder_paths
            out_root = folder_paths.get_output_directory()
        except Exception:
            # Fallback: ComfyUI conventional layout
            out_root = os.path.abspath("output")

        # Sanitize subdirectory and filename to prevent traversal
        safe_sub = re.sub(r"[^A-Za-z0-9_\-/]", "_", (subdirectory or "").strip("/\\"))
        safe_prefix = re.sub(r"[^A-Za-z0-9_\-]", "_", (filename_prefix or "audio").strip())
        if not safe_prefix:
            safe_prefix = "audio"
        if ".." in safe_sub or os.path.isabs(safe_sub):
            safe_sub = ""

        out_dir = os.path.join(out_root, safe_sub) if safe_sub else out_root
        out_dir = os.path.abspath(out_dir)
        # Final containment check
        if not out_dir.startswith(os.path.abspath(out_root)):
            raise ValueError("subdirectory escapes the output directory.")

        os.makedirs(out_dir, exist_ok=True)

        # Find a non-colliding filename
        ts = int(time.time())
        for i in range(0, 10000):
            fname = f"{safe_prefix}_{ts}_{i:04d}.{format}"
            full = os.path.join(out_dir, fname)
            if not os.path.exists(full):
                break

        wf = audio["waveform"]
        sr = int(audio["sample_rate"])
        if wf.dim() == 3: wf = wf[0]
        if wf.dim() == 1: wf = wf.unsqueeze(0)
        data = wf.detach().cpu().float().numpy().T  # (samples, channels)

        try:
            import soundfile as sf
            sf.write(full, data, sr, format=format.upper())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write {format} (need soundfile / libsndfile installed): {exc}"
            )

        print(f"[ElevenLabs Pro] Audio saved to {full}")
        return (full,)


# ============================================================
# Registration
# ============================================================

NODE_CLASS_MAPPINGS = {
    "ElevenLabsPro_APIKey": ElevenLabsPro_APIKey,
    "ElevenLabsPro_VoiceSelector": ElevenLabsPro_VoiceSelector,
    "ElevenLabsPro_FetchVoices": ElevenLabsPro_FetchVoices,
    "ElevenLabsPro_GetVoiceByName": ElevenLabsPro_GetVoiceByName,
    "ElevenLabsPro_VoiceClone": ElevenLabsPro_VoiceClone,
    "ElevenLabsPro_VoiceDesign": ElevenLabsPro_VoiceDesign,
    "ElevenLabsPro_VoiceCreate": ElevenLabsPro_VoiceCreate,
    "ElevenLabsPro_TTS": ElevenLabsPro_TTS,
    "ElevenLabsPro_TTSTimestamps": ElevenLabsPro_TTSTimestamps,
    "ElevenLabsPro_STS": ElevenLabsPro_STS,
    "ElevenLabsPro_SFX": ElevenLabsPro_SFX,
    "ElevenLabsPro_AudioIsolation": ElevenLabsPro_AudioIsolation,
    "ElevenLabsPro_STT": ElevenLabsPro_STT,
    "ElevenLabsPro_Dialogue": ElevenLabsPro_Dialogue,
    "ElevenLabsPro_Music": ElevenLabsPro_Music,
    "ElevenLabsPro_AccountInfo": ElevenLabsPro_AccountInfo,
    # New utility nodes (v2.1)
    "ElevenLabsPro_VoiceTagInserter": ElevenLabsPro_VoiceTagInserter,
    "ElevenLabsPro_SubtitleExport": ElevenLabsPro_SubtitleExport,
    "ElevenLabsPro_AudioConcat": ElevenLabsPro_AudioConcat,
    "ElevenLabsPro_AudioNormalize": ElevenLabsPro_AudioNormalize,
    "ElevenLabsPro_AudioMetadata": ElevenLabsPro_AudioMetadata,
    "ElevenLabsPro_AudioChannels": ElevenLabsPro_AudioChannels,
    "ElevenLabsPro_AudioTrim": ElevenLabsPro_AudioTrim,
    "ElevenLabsPro_CostEstimator": ElevenLabsPro_CostEstimator,
    "ElevenLabsPro_VoiceSettingsPreset": ElevenLabsPro_VoiceSettingsPreset,
    "ElevenLabsPro_SaveAudio": ElevenLabsPro_SaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenLabsPro_APIKey": "ElevenLabs Pro - API Key",
    "ElevenLabsPro_VoiceSelector": "ElevenLabs Pro - Voice Selector",
    "ElevenLabsPro_FetchVoices": "ElevenLabs Pro - Fetch Voices",
    "ElevenLabsPro_GetVoiceByName": "ElevenLabs Pro - Get Voice By Name",
    "ElevenLabsPro_VoiceClone": "ElevenLabs Pro - Voice Clone",
    "ElevenLabsPro_VoiceDesign": "ElevenLabs Pro - Voice Design",
    "ElevenLabsPro_VoiceCreate": "ElevenLabs Pro - Voice Create",
    "ElevenLabsPro_TTS": "ElevenLabs Pro - Text to Speech",
    "ElevenLabsPro_TTSTimestamps": "ElevenLabs Pro - TTS with Timestamps",
    "ElevenLabsPro_STS": "ElevenLabs Pro - Speech to Speech",
    "ElevenLabsPro_SFX": "ElevenLabs Pro - Sound Effects",
    "ElevenLabsPro_AudioIsolation": "ElevenLabs Pro - Audio Isolation",
    "ElevenLabsPro_STT": "ElevenLabs Pro - Speech to Text",
    "ElevenLabsPro_Dialogue": "ElevenLabs Pro - Text to Dialogue",
    "ElevenLabsPro_Music": "ElevenLabs Pro - Music Generation",
    "ElevenLabsPro_AccountInfo": "ElevenLabs Pro - Account Info",
    "ElevenLabsPro_VoiceTagInserter": "ElevenLabs Pro - Voice Tag Inserter",
    "ElevenLabsPro_SubtitleExport": "ElevenLabs Pro - Subtitle Export (SRT/VTT)",
    "ElevenLabsPro_AudioConcat": "ElevenLabs Pro - Audio Concat",
    "ElevenLabsPro_AudioNormalize": "ElevenLabs Pro - Audio Normalize",
    "ElevenLabsPro_AudioMetadata": "ElevenLabs Pro - Audio Metadata",
    "ElevenLabsPro_AudioChannels": "ElevenLabs Pro - Audio Channels",
    "ElevenLabsPro_AudioTrim": "ElevenLabs Pro - Audio Trim",
    "ElevenLabsPro_CostEstimator": "ElevenLabs Pro - Cost Estimator",
    "ElevenLabsPro_VoiceSettingsPreset": "ElevenLabs Pro - Voice Settings Preset",
    "ElevenLabsPro_SaveAudio": "ElevenLabs Pro - Save Audio",
}
