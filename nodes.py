"""
ElevenLabs nodes for ComfyUI-ElevenLabs-Pro.

15 nodes total:
  Voice:   VoiceSelector, FetchVoices, VoiceClone, VoiceDesign, VoiceCreate
  TTS:     TTS, TTSTimestamps, Dialogue
  Audio:   STS, SFX, AudioIsolation, STT
  Music:   Music
  Config:  APIKey, AccountInfo
"""

import json

from .shared.node_utils import AlwaysExecuteMixin

from .utils import (
    ELEVENLABS_API_BASE,
    TTS_MODELS,
    STS_MODELS,
    STT_MODELS,
    SFX_MODELS,
    OUTPUT_FORMATS,
    LANGUAGE_OPTIONS,
    LANGUAGE_MAP,
    VOICE_OPTIONS,
    VOICE_MAP,
    get_api_key,
    api_headers,
    api_post,
    api_get,
    check_response,
    fetch_all_voices,
    validate_text_length,
    audio_bytes_to_comfy,
    comfy_audio_to_bytes,
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

CATEGORY_BASE = "ElevenLabs Pro"


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
    """Fetch all voices from your ElevenLabs account (library + cloned + generated)."""

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
                "force_refresh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Bypass voice cache and fetch fresh from API.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("voices_json", "voice_id_list",)
    FUNCTION = "fetch"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def fetch(self, api_key, search="", category="all", force_refresh=False):
        key = get_api_key(api_key)
        voices = fetch_all_voices(key, force_refresh=force_refresh)

        # E6: search also checks description and use_case
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

        voices_json = json.dumps(voices, indent=2)
        id_list = "\n".join(f"{v['voice_id']}  # {v['name']}" for v in voices)
        return (voices_json, id_list,)


class ElevenLabsPro_VoiceClone(AlwaysExecuteMixin):
    """Create an instant voice clone from audio samples (up to 8 files)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "voice_name": ("STRING", {"default": "My Cloned Voice", "tooltip": "Name for the cloned voice."}),
                "audio1": ("AUDIO",),
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "clone"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def clone(self, api_key, voice_name, audio1,
              audio2=None, audio3=None, audio4=None,
              audio5=None, audio6=None, audio7=None, audio8=None,
              description="", remove_background_noise=False):
        key = get_api_key(api_key)

        files = []
        for i, audio in enumerate([audio1, audio2, audio3, audio4,
                                    audio5, audio6, audio7, audio8]):
            if audio is None:
                continue
            audio_bytes, _ = comfy_audio_to_bytes(audio)
            files.append(("files", (f"sample_{i+1}.wav", audio_bytes, "audio/wav")))

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
        check_response(resp)
        voice_id = resp.json().get("voice_id", "")
        print(f"[ElevenLabs Pro] Voice cloned: {voice_id} ({voice_name})")
        return (voice_id,)


class ElevenLabsPro_VoiceDesign(AlwaysExecuteMixin):
    """Design a new voice from a text prompt and voice description."""

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
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("preview_audio", "generated_voice_id",)
    FUNCTION = "design"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def design(self, api_key, text, voice_description):
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
        check_response(resp)
        result = resp.json()

        previews = result.get("previews", [])
        if not previews:
            raise RuntimeError("No voice previews returned by API.")

        first = previews[0]
        generated_voice_id = first.get("generated_voice_id", "")

        # The preview audio is base64-encoded
        import base64
        audio_b64 = first.get("audio_base_64", "")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            audio = audio_bytes_to_comfy(audio_bytes, "mp3_44100_128")
        else:
            # Fallback: empty audio
            import torch
            audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}

        return (audio, generated_voice_id,)


class ElevenLabsPro_VoiceCreate(AlwaysExecuteMixin):
    """Save a designed voice preview as a permanent voice in your library."""

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
            },
            "optional": {
                "voice_description": ("STRING", {
                    "default": "",
                    "tooltip": "Optional description for the saved voice.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "create"
    CATEGORY = f"{CATEGORY_BASE}/Voice"

    def create(self, api_key, generated_voice_id, voice_name, voice_description=""):
        if not generated_voice_id or not generated_voice_id.strip():
            raise ValueError("generated_voice_id is required. Connect from Voice Design node.")

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
        check_response(resp)
        voice_id = resp.json().get("voice_id", "")
        print(f"[ElevenLabs Pro] Voice created from preview: {voice_id} ({voice_name})")
        return (voice_id,)


# ============================================================
# Text to Speech
# ============================================================

class ElevenLabsPro_TTS(AlwaysExecuteMixin):
    """Full-featured Text to Speech with all ElevenLabs API parameters."""

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
                    "tooltip": "Language for the model. Auto Detect lets the model decide.",
                }),
                "apply_text_normalization": (["auto", "on", "off"], {
                    "default": "auto",
                    "tooltip": "Text normalization: 'auto' = model decides, 'on' = always normalize, 'off' = skip.",
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
                    "tooltip": "Text that came BEFORE this segment. Helps maintain natural flow across chunks.",
                }),
                "next_text": ("STRING", {
                    "default": "",
                    "tooltip": "Text that comes AFTER this segment. Helps the model anticipate pacing.",
                }),
                "use_pvc_as_ivc": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use IVC version of a Professional Voice Clone.",
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
                 use_pvc_as_ivc=False,
                 pronunciation_dictionary_locators=""):

        if not text or not text.strip():
            raise ValueError("Text input is empty.")
        if not voice_id or not voice_id.strip():
            raise ValueError("voice_id is required. Connect a Voice Selector node.")

        validate_text_length(text, model)

        key = get_api_key(api_key)
        lang_code = LANGUAGE_MAP.get(language, "")

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
            "apply_text_normalization": apply_text_normalization,
        }

        if lang_code:
            body["language_code"] = lang_code
        if seed > 0:
            body["seed"] = seed
        if previous_text and previous_text.strip():
            body["previous_text"] = previous_text.strip()
        if next_text and next_text.strip():
            body["next_text"] = next_text.strip()
        if use_pvc_as_ivc:
            body["use_pvc_as_ivc"] = True

        # E13: pronunciation dictionary locators
        if pronunciation_dictionary_locators and pronunciation_dictionary_locators.strip():
            try:
                locators = json.loads(pronunciation_dictionary_locators.strip())
                if isinstance(locators, list) and locators:
                    body["pronunciation_dictionary_locators"] = locators
            except json.JSONDecodeError:
                pass  # Silently ignore malformed JSON

        params = {"output_format": output_format}

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-speech/{voice_id.strip()}",
            key,
            timeout=300,
            json=body,
            params=params,
            headers=api_headers(key),
        )
        check_response(resp)

        audio_bytes = resp.content
        info = json.dumps({
            "model": model,
            "voice_id": voice_id.strip(),
            "format": output_format,
            "bytes": len(audio_bytes),
            "characters": len(text),
        })

        audio = audio_bytes_to_comfy(audio_bytes, output_format)
        return (audio, info,)


class ElevenLabsPro_TTSTimestamps(AlwaysExecuteMixin):
    """Text to Speech with word-level timestamps returned as JSON."""

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
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "timestamps_json",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY_BASE}/TTS"

    def generate(self, api_key, text, voice_id, model,
                 stability=0.5, similarity_boost=0.75,
                 output_format="mp3_44100_128", seed=0):

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
        }
        if seed > 0:
            body["seed"] = seed

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-speech/{voice_id.strip()}/with-timestamps",
            key,
            timeout=300,
            json=body,
            params={"output_format": output_format},
            headers=api_headers(key),
        )
        check_response(resp)

        result = resp.json()

        # Audio is base64-encoded in the response
        import base64
        audio_b64 = result.get("audio_base_64", "")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            audio = audio_bytes_to_comfy(audio_bytes, output_format)
        else:
            import torch
            audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}

        alignment = result.get("alignment", {})
        timestamps_json = json.dumps(alignment, indent=2)

        return (audio, timestamps_json,)


# ============================================================
# Speech to Speech
# ============================================================

class ElevenLabsPro_STS(AlwaysExecuteMixin):
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
                "output_format": (OUTPUT_FORMATS, {
                    "default": "mp3_44100_128",
                    "tooltip": _TT_OUTPUT_FMT,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": _TT_SEED,
                }),
                "remove_background_noise": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "transform"
    CATEGORY = f"{CATEGORY_BASE}/Audio"

    def transform(self, api_key, audio, voice_id, model,
                  stability=0.5, similarity_boost=0.75, style=0.0, speed=1.0,
                  use_speaker_boost=True, output_format="mp3_44100_128",
                  seed=0, remove_background_noise=False):

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
            "remove_background_noise": str(remove_background_noise).lower(),
        }
        if seed > 0:
            data["seed"] = str(seed)

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/speech-to-speech/{voice_id.strip()}",
            key,
            timeout=300,
            headers={"xi-api-key": key},
            data=data,
            files={"audio": ("input.wav", audio_bytes, "audio/wav")},
            params={"output_format": output_format},
        )
        check_response(resp)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Sound Effects
# ============================================================

class ElevenLabsPro_SFX(AlwaysExecuteMixin):
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
                    "default": "eleven_sfx_v2",
                    "tooltip": "Sound effects model to use.",
                }),
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 30.0, "step": 0.1,
                    "tooltip": "Duration in seconds.",
                }),
                "prompt_influence": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How closely generation follows the prompt.",
                }),
                "loop": ("BOOLEAN", {"default": False, "tooltip": "Create a smoothly looping sound effect."}),
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

    def generate(self, api_key, text, model="eleven_sfx_v2",
                 duration=5.0, prompt_influence=0.3,
                 loop=False, output_format="mp3_44100_128", seed=0):

        if not text or not text.strip():
            raise ValueError("Text description is empty.")

        key = get_api_key(api_key)

        body = {
            "text": text,
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
        check_response(resp)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Audio Isolation
# ============================================================

class ElevenLabsPro_AudioIsolation(AlwaysExecuteMixin):
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
        )
        check_response(resp)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Speech to Text
# ============================================================

class ElevenLabsPro_STT(AlwaysExecuteMixin):
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
                "timestamps_granularity": (["word", "character", "none"], {"default": "word"}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": _TT_SEED,
                }),
                "diarization_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Threshold for speaker diarization. Higher = stricter speaker separation.",
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
                    "tooltip": "When True, removes filler words and stutters from transcript.",
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

        data = {"model_id": model}
        if language_code and language_code.strip():
            data["language_code"] = language_code.strip()
        if tag_audio_events:
            data["tag_audio_events"] = "true"
        if diarize:
            data["diarize"] = "true"
        if num_speakers > 0:
            data["num_speakers"] = str(num_speakers)
        if timestamps_granularity != "none":
            data["timestamps_granularity"] = timestamps_granularity
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
        check_response(resp)

        result = resp.json()
        text = result.get("text", "")
        lang = result.get("language_code", "")
        words = json.dumps(result.get("words", []), indent=2)
        return (text, lang, words,)


# ============================================================
# Dialogue (multi-speaker, v3)
# ============================================================

class ElevenLabsPro_Dialogue(AlwaysExecuteMixin):
    """Generate multi-speaker dialogue. Each speaker gets their own voice and text (up to 10)."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "text1": ("STRING", {"multiline": True, "default": "", "tooltip": "Speaker 1 text."}),
                "voice_id1": ("STRING", {"default": "", "tooltip": "Speaker 1 voice ID."}),
            },
            "optional": {},
        }

        # Speakers 2-10
        for i in range(2, 11):
            inputs["optional"][f"text{i}"] = ("STRING", {"multiline": True, "default": ""})
            inputs["optional"][f"voice_id{i}"] = ("STRING", {"default": ""})

        inputs["optional"].update({
            "stability": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": _TT_STABILITY,
            }),
            "apply_text_normalization": (["auto", "on", "off"], {"default": "auto"}),
            "language": (LANGUAGE_OPTIONS, {"default": "Auto Detect"}),
            "output_format": (["mp3_44100_192", "opus_48000_192"], {
                "default": "mp3_44100_192",
                "tooltip": _TT_OUTPUT_FMT,
            }),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 4294967295,
                "tooltip": _TT_SEED,
            }),
        })
        return inputs

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY_BASE}/TTS"

    def generate(self, api_key, text1, voice_id1, **kwargs):
        key = get_api_key(api_key)
        lang_code = LANGUAGE_MAP.get(kwargs.get("language", "Auto Detect"), "")

        inputs = []
        # Collect all text/voice pairs (1-10)
        all_pairs = [(text1, voice_id1)]
        for i in range(2, 11):
            t = kwargs.get(f"text{i}", "")
            v = kwargs.get(f"voice_id{i}", "")
            all_pairs.append((t, v))

        for text, vid in all_pairs:
            if text and text.strip() and vid and vid.strip():
                inputs.append({"text": text.strip(), "voice_id": vid.strip()})

        if not inputs:
            raise ValueError("At least one text + voice_id pair is required.")

        stability = kwargs.get("stability", 0.5)
        apply_text_normalization = kwargs.get("apply_text_normalization", "auto")
        output_format = kwargs.get("output_format", "mp3_44100_192")
        seed = kwargs.get("seed", 0)

        body = {
            "inputs": inputs,
            "model_id": "eleven_v3",
            "settings": {"stability": stability},
            "apply_text_normalization": apply_text_normalization,
        }
        if lang_code:
            body["language_code"] = lang_code
        if seed > 0:
            body["seed"] = seed

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/text-to-dialogue",
            key,
            timeout=300,
            json=body,
            params={"output_format": output_format},
            headers=api_headers(key),
        )
        check_response(resp)

        return (audio_bytes_to_comfy(resp.content, output_format),)


# ============================================================
# Music Generation
# ============================================================

class ElevenLabsPro_Music(AlwaysExecuteMixin):
    """Generate music from a text prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text description of the music to generate.",
                }),
            },
            "optional": {
                "duration_seconds": ("FLOAT", {
                    "default": 30.0, "min": 5.0, "max": 300.0, "step": 1.0,
                    "tooltip": "Duration of generated music in seconds.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 4294967295,
                    "tooltip": _TT_SEED,
                }),
                "force_instrumental": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate instrumental-only music (no vocals).",
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

    def generate(self, api_key, prompt,
                 duration_seconds=30.0, seed=0,
                 force_instrumental=False, output_format="mp3_44100_128"):

        if not prompt or not prompt.strip():
            raise ValueError("Music prompt is empty.")

        key = get_api_key(api_key)

        body = {
            "prompt": prompt.strip(),
            "duration_seconds": duration_seconds,
        }
        if seed > 0:
            body["seed"] = seed
        if force_instrumental:
            body["force_instrumental"] = True

        resp = api_post(
            f"{ELEVENLABS_API_BASE}/v1/music",
            key,
            timeout=600,
            json=body,
            params={"output_format": output_format},
            headers=api_headers(key),
        )
        check_response(resp)

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

    def passthrough(self, api_key):
        key = get_api_key(api_key)
        return (key,)


class ElevenLabsPro_AccountInfo(AlwaysExecuteMixin):
    """Retrieve ElevenLabs account info: subscription tier, character usage."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT",)
    RETURN_NAMES = ("account_json", "subscription_tier", "character_count", "character_limit",)
    FUNCTION = "fetch"
    CATEGORY = f"{CATEGORY_BASE}/Config"

    def fetch(self, api_key):
        key = get_api_key(api_key)

        resp = api_get(
            f"{ELEVENLABS_API_BASE}/v1/user",
            key,
        )
        check_response(resp)

        data = resp.json()
        sub = data.get("subscription", {})
        tier = sub.get("tier", "unknown")
        char_count = sub.get("character_count", 0)
        char_limit = sub.get("character_limit", 0)

        account_json = json.dumps(data, indent=2)
        return (account_json, tier, char_count, char_limit,)


# ============================================================
# Registration
# ============================================================

NODE_CLASS_MAPPINGS = {
    "ElevenLabsPro_APIKey": ElevenLabsPro_APIKey,
    "ElevenLabsPro_VoiceSelector": ElevenLabsPro_VoiceSelector,
    "ElevenLabsPro_FetchVoices": ElevenLabsPro_FetchVoices,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenLabsPro_APIKey": "ElevenLabs - API Key",
    "ElevenLabsPro_VoiceSelector": "ElevenLabs - Voice Selector",
    "ElevenLabsPro_FetchVoices": "ElevenLabs - Fetch Voices",
    "ElevenLabsPro_VoiceClone": "ElevenLabs - Voice Clone",
    "ElevenLabsPro_VoiceDesign": "ElevenLabs - Voice Design",
    "ElevenLabsPro_VoiceCreate": "ElevenLabs - Voice Create",
    "ElevenLabsPro_TTS": "ElevenLabs - Text to Speech",
    "ElevenLabsPro_TTSTimestamps": "ElevenLabs - TTS with Timestamps",
    "ElevenLabsPro_STS": "ElevenLabs - Speech to Speech",
    "ElevenLabsPro_SFX": "ElevenLabs - Sound Effects",
    "ElevenLabsPro_AudioIsolation": "ElevenLabs - Audio Isolation",
    "ElevenLabsPro_STT": "ElevenLabs - Speech to Text",
    "ElevenLabsPro_Dialogue": "ElevenLabs - Text to Dialogue",
    "ElevenLabsPro_Music": "ElevenLabs - Music Generation",
    "ElevenLabsPro_AccountInfo": "ElevenLabs - Account Info",
}
