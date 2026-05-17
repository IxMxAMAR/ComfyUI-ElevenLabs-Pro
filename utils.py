"""
ElevenLabs API client -- constants, voice cache, API helpers, audio conversion.

All HTTP calls go through shared.retry.api_request_with_retry for automatic
exponential-backoff on transient errors (429 / 5xx). Multipart file pointers
are seeked back to 0 between attempts so retries don't send empty bodies.

Secrets are scrubbed from error messages before they propagate to ComfyUI
logs/UI/workflow PNGs via shared.errors.parse_error_response(extra_secrets=).
"""

import io
import os
import time
import hashlib
import struct
from collections import OrderedDict

import numpy as np
import torch

try:
    from .shared.retry import api_request_with_retry
    from .shared.errors import APIPermanentError, parse_error_response, scrub_secrets
except ImportError:
    from shared.retry import api_request_with_retry
    from shared.errors import APIPermanentError, parse_error_response, scrub_secrets

# ============================================================
# Constants
# ============================================================

ELEVENLABS_API_BASE = "https://api.elevenlabs.io"

# Verified from /v1/models endpoint. See README for character limits.
TTS_MODELS = [
    "eleven_v3",                  # Latest flagship, expressive, supports v3 audio tags
    "eleven_multilingual_v2",     # Stable multilingual, ~32 languages
    "eleven_flash_v2_5",          # Fast multilingual (lowest latency)
    "eleven_turbo_v2_5",          # Multilingual turbo (better than flash, faster than v2)
    "eleven_flash_v2",            # English-only fast
    "eleven_turbo_v2",            # English-only turbo (legacy)
    "eleven_multilingual_v1",     # Legacy multilingual
    "eleven_monolingual_v1",      # Legacy English (Eleven v1)
]

# Models which REQUIRE an explicit language_code (won't auto-detect)
TTS_MODELS_REQUIRING_LANGUAGE = {
    "eleven_turbo_v2_5",
    "eleven_flash_v2_5",
}

# Models for which apply_text_normalization MUST be "off" per ElevenLabs spec
TTS_MODELS_NORMALIZATION_OFF = {"eleven_v3"}

STS_MODELS = [
    "eleven_multilingual_sts_v2",
    "eleven_english_sts_v2",
]

# scribe_v1 + scribe_v2 (verified 2026-05 against docs)
STT_MODELS = [
    "scribe_v2",
    "scribe_v1",
]

# Correct ID is `eleven_text_to_sound_v2` (verified 2026-05 against docs).
# `eleven_sfx_v2` was an internal alias kept as a fallback for backward compat.
SFX_MODELS = [
    "eleven_text_to_sound_v2",
    "eleven_sfx_v2",
]

# Verified 2026-05 against /v1/music docs.
MUSIC_MODELS = [
    "music_v1",
]

# Per-endpoint max durations (seconds)
SFX_MAX_DURATION = 30.0    # Verified 0.5–30s per /v1/text-to-sound-effects docs
MUSIC_MAX_SECONDS = 600.0  # Verified 3–600s per /v1/music docs

# User-facing list (ulaw_8000 and alaw_8000 removed)
OUTPUT_FORMATS = [
    "mp3_44100_192",
    "mp3_44100_128",
    "mp3_44100_96",
    "mp3_44100_64",
    "mp3_44100_32",
    "mp3_24000_48",
    "mp3_22050_32",
    "wav_44100",
    "wav_48000",
    "wav_32000",
    "wav_24000",
    "wav_22050",
    "wav_16000",
    "wav_8000",
    "pcm_44100",
    "pcm_48000",
    "pcm_32000",
    "pcm_24000",
    "pcm_22050",
    "pcm_16000",
    "pcm_8000",
    "opus_48000_192",
    "opus_48000_128",
    "opus_48000_96",
    "opus_48000_64",
    "opus_48000_32",
]

# Formats that ComfyUI can decode into audio tensors
DECODABLE_FORMATS = {
    "mp3_44100_192", "mp3_44100_128", "mp3_44100_96", "mp3_44100_64",
    "mp3_44100_32", "mp3_24000_48", "mp3_22050_32",
    "wav_44100", "wav_48000", "wav_32000", "wav_24000",
    "wav_22050", "wav_16000", "wav_8000",
    "pcm_44100", "pcm_48000", "pcm_32000", "pcm_24000",
    "pcm_22050", "pcm_16000", "pcm_8000",
}

LANGUAGES = [
    ("Auto Detect", ""),
    ("English", "en"),
    ("Arabic", "ar"),
    ("Bulgarian", "bg"),
    ("Chinese", "zh"),
    ("Croatian", "hr"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("Filipino", "fil"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("German", "de"),
    ("Greek", "el"),
    ("Hindi", "hi"),
    ("Hungarian", "hu"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Malay", "ms"),
    ("Norwegian", "no"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Romanian", "ro"),
    ("Russian", "ru"),
    ("Slovak", "sk"),
    ("Spanish", "es"),
    ("Swedish", "sv"),
    ("Tamil", "ta"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
    ("Vietnamese", "vi"),
]

LANGUAGE_OPTIONS = [f"{name} ({code})" if code else name for name, code in LANGUAGES]
LANGUAGE_MAP = {f"{name} ({code})" if code else name: code for name, code in LANGUAGES}

# Extended predefined voices (all default ElevenLabs library voices)
PREDEFINED_VOICES = [
    ("21m00Tcm4TlvDq8ikWAM", "Rachel", "female", "american"),
    ("29vD33N1CtxCmqQRPOHJ", "Drew", "male", "american"),
    ("2EiwWnXFnvU5JabPnv8n", "Clyde", "male", "american"),
    ("5Q0t7uMcjvnagumLfvZi", "Paul", "male", "american"),
    ("AZnzlk1XvdvUeBnXmlld", "Domi", "female", "american"),
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger", "male", "american"),
    ("EXAVITQu4vr4xnSDxMaL", "Sarah", "female", "american"),
    ("ErXwobaYiN019PkySvjV", "Antoni", "male", "american"),
    ("FGY2WhTYpPnrIDTdsKH5", "Laura", "female", "american"),
    ("GBv7mTt0atIp3Br8iCZE", "Thomas", "male", "american"),
    ("IKne3meq5aSn9XLyUdCD", "Charlie", "male", "australian"),
    ("JBFqnCBsd6RMkjVDRZzb", "George", "male", "british"),
    ("MF3mGyEYCl7XYWbV9V6O", "Elli", "female", "american"),
    ("N2lVS1w4EtoT3dr4eOWO", "Callum", "male", "american"),
    ("SAz9YHcvj6GT2YYXdXww", "River", "nonbinary", "american"),
    ("SOYHLrjzK2X1ezoPC6cr", "Harry", "male", "american"),
    ("TX3LPaxmHKxFdv7VOQHJ", "Liam", "male", "american"),
    ("ThT5KcBeYPX3keUQqHPh", "Dorothy", "female", "british"),
    ("TxGEqnHWrfWFTfGW9XjX", "Josh", "male", "american"),
    ("VR6AewLTigWG4xSOukaG", "Arnold", "male", "american"),
    ("Xb7hH8MSUJpSbSDYk0k2", "Alice", "female", "british"),
    ("XrExE9yKIg1WjnnlVkGX", "Matilda", "female", "american"),
    ("Zlb1dXrM653N07WRdFW3", "Joseph", "male", "british"),
    ("bIHbv24MWmeRgasZH58o", "Will", "male", "american"),
    ("cgSgspJ2msm6clMCkdW9", "Jessica", "female", "american"),
    ("cjVigY5qzO86Huf0OWal", "Eric", "male", "american"),
    ("g5CIjZEefAph4nQFvHAz", "Ethan", "male", "american"),
    ("hpp4J3VqNfWAUOO0d1Us", "Bella", "female", "american"),
    ("iP95p4xoKVk53GoZ742B", "Chris", "male", "american"),
    ("jBpfuIE2acCO8z3wKNLl", "Gigi", "female", "american"),
    ("jsCqWAovK2LkecY7zXl4", "Freya", "female", "american"),
    ("nPczCjzI2devNBz1zQrb", "Brian", "male", "american"),
    ("oWAxZDx7w5VEj9dCyTzz", "Grace", "female", "american"),
    ("onwK4e9ZLuTAKqWW03F9", "Daniel", "male", "british"),
    ("pFZP5JQG7iQjIQuC4Bku", "Lily", "female", "british"),
    ("pNInz6obpgDQGcFmaJgB", "Adam", "male", "american"),
    ("pqHfZKP75CvOlQylNhV4", "Bill", "male", "american"),
    ("t0jbNlBVZ17f02VDIeMI", "Emily", "female", "american"),
    ("yoZ06aMxZJJ28mfd3POQ", "Sam", "male", "american"),
    ("z9fAnlkpzviPz146aGWa", "Glinda", "female", "american"),
    ("zcAOhNBS3c14rBihAFp1", "Giovanni", "male", "italian"),
    ("zrHiDhphv9ZnVXBqCLjz", "Mimi", "female", "swedish"),
]

VOICE_OPTIONS = ["(Custom voice_id)"] + [
    f"{name} ({gender}, {accent})" for _, name, gender, accent in PREDEFINED_VOICES
]
VOICE_MAP = {
    f"{name} ({gender}, {accent})": vid
    for vid, name, gender, accent in PREDEFINED_VOICES
}

# Model character limits
_MODEL_CHAR_LIMITS = {
    "eleven_v3": 5000,
    "eleven_multilingual_v2": 10000,
    "eleven_flash_v2_5": 40000,
    "eleven_turbo_v2_5": 40000,
    "eleven_flash_v2": 40000,
    "eleven_turbo_v2": 30000,
    "eleven_multilingual_v1": 10000,
    "eleven_monolingual_v1": 5000,
}

# v3 expression / audio tags (for VoiceTagInserter helper). Names from
# ElevenLabs v3 docs; not exhaustive.
V3_AUDIO_TAGS = [
    "whispers", "laughs", "sighs", "exhales", "inhales",
    "excited", "sad", "angry", "happy", "scared",
    "shouts", "mumbles", "groans", "coughs", "sneezes",
    "yawns", "cries", "giggles", "chuckles", "snorts",
    "gasps", "screams", "cheers", "sarcastic", "curious",
    "nervous", "calm", "tired", "exhausted",
]

# ============================================================
# Voice Cache  (bounded LRU; capped per-process to prevent memory growth
# across long-running ComfyUI sessions that touch many API keys)
# ============================================================

_VOICE_CACHE_TTL = 300       # 5 minutes
_VOICE_CACHE_MAX_KEYS = 32   # Cap distinct api_key hashes
_voice_cache: "OrderedDict[str, list]" = OrderedDict()
_voice_cache_time: dict = {}


def _cache_set(key: str, value):
    if key in _voice_cache:
        _voice_cache.move_to_end(key)
    _voice_cache[key] = value
    _voice_cache_time[key] = time.time()
    while len(_voice_cache) > _VOICE_CACHE_MAX_KEYS:
        oldest, _ = _voice_cache.popitem(last=False)
        _voice_cache_time.pop(oldest, None)


def _cache_get(key: str):
    if key not in _voice_cache:
        return None
    if (time.time() - _voice_cache_time.get(key, 0)) >= _VOICE_CACHE_TTL:
        _voice_cache.pop(key, None)
        _voice_cache_time.pop(key, None)
        return None
    _voice_cache.move_to_end(key)
    return _voice_cache[key]


# ============================================================
# API Helpers
# ============================================================


def get_api_key(api_key_input: str = "") -> str:
    """Resolve an ElevenLabs API key from input or ELEVENLABS_API_KEY env var."""
    key = api_key_input.strip() if api_key_input else ""
    if key:
        return key
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if key:
        return key
    raise ValueError(
        "ElevenLabs API key required. Either:\n"
        "  1. Connect an API key string to the node\n"
        "  2. Set ELEVENLABS_API_KEY environment variable"
    )


def api_headers(api_key: str) -> dict:
    """Return standard ElevenLabs request headers."""
    return {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }


def _scrub_and_reraise(exc, api_key):
    """Re-raise `exc` with all instances of `api_key` (and key-like patterns)
    scrubbed from its message. Preserves exception type and APIError attrs.
    """
    try:
        from .shared.errors import APIError
    except ImportError:
        from shared.errors import APIError
    raw = str(exc)
    scrubbed = scrub_secrets(raw, extra_secrets=[api_key])
    if scrubbed == raw:
        raise exc
    if isinstance(exc, APIError):
        # Rebuild with scrubbed detail (use .detail not str(exc) to avoid the
        # prefix "[Service] HTTP 400: " getting double-stuffed into the new detail).
        new_detail = scrub_secrets(exc.detail, extra_secrets=[api_key])
        new = type(exc)(exc.service, exc.status_code, new_detail)
        raise new from exc.__cause__
    # Generic: wrap in a fresh exception of the same type if possible
    try:
        raise type(exc)(scrubbed) from exc
    except TypeError:
        # Couldn't construct — fall back to RuntimeError
        raise RuntimeError(scrubbed) from exc


def api_post(url: str, api_key: str, timeout: float = 300, **kwargs) -> "requests.Response":
    """POST with retry logic. Accepts the same kwargs as requests
    (json, data, files, params, headers).

    Scrubs the api_key from any error message before raising so the key
    cannot leak into ComfyUI logs / UI / saved workflow PNGs.
    """
    headers = kwargs.pop("headers", None) or {"xi-api-key": api_key}
    if "Content-Type" not in headers and "files" not in kwargs and "data" not in kwargs:
        headers["Content-Type"] = "application/json"
    try:
        return api_request_with_retry(
            "POST",
            url,
            headers=headers,
            timeout=timeout,
            service_name="ElevenLabs",
            **kwargs,
        )
    except Exception as exc:
        _scrub_and_reraise(exc, api_key)


def api_get(url: str, api_key: str, timeout: float = 30, **kwargs) -> "requests.Response":
    """GET with retry logic."""
    headers = kwargs.pop("headers", None) or {"xi-api-key": api_key}
    try:
        return api_request_with_retry(
            "GET",
            url,
            headers=headers,
            timeout=timeout,
            service_name="ElevenLabs",
            **kwargs,
        )
    except Exception as exc:
        _scrub_and_reraise(exc, api_key)


def check_response(resp, api_key: str = "") -> None:
    """Raise APIError with parsed message if status >= 400.

    `api_key` is scrubbed from the detail message before construction so
    even a misbehaving upstream that echoes the header can't leak the key.
    Note: api_request_with_retry already raises through parse_error_response
    on non-2xx responses, so this is mostly defensive for callers that
    bypass that path (none currently).
    """
    if resp.status_code >= 400:
        raise parse_error_response(
            "ElevenLabs", resp.status_code, resp,
            extra_secrets=[api_key] if api_key else None,
        )


def fetch_all_voices(api_key: str, force_refresh: bool = False) -> list:
    """Paginated v2 voices with 5-min bounded LRU cache."""
    cache_key = hashlib.sha256(api_key.encode()).hexdigest()[:16]

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    voices = []
    next_token = None
    max_pages = 100  # Defensive: cap pagination at ~10k voices
    pages = 0

    while pages < max_pages:
        params: dict = {"page_size": 100}
        if next_token:
            params["next_page_token"] = next_token

        resp = api_get(
            f"{ELEVENLABS_API_BASE}/v2/voices",
            api_key,
            params=params,
        )
        data = resp.json()

        for v in data.get("voices", []):
            labels = v.get("labels", {}) or {}
            voices.append({
                "voice_id": v["voice_id"],
                "name": v.get("name", "Unknown"),
                "category": v.get("category", ""),
                "gender": labels.get("gender", ""),
                "accent": labels.get("accent", ""),
                "age": labels.get("age", ""),
                "description": labels.get("description", ""),
                "use_case": labels.get("use_case", ""),
            })

        if not data.get("has_more"):
            break
        next_token = data.get("next_page_token")
        if not next_token:
            # Defensive: API said has_more=True but gave no token. Stop.
            break
        pages += 1

    _cache_set(cache_key, voices)
    return voices


def validate_text_length(text: str, model: str) -> None:
    """Raise ValueError if text exceeds model character limit."""
    limit = _MODEL_CHAR_LIMITS.get(model)
    if limit and len(text) > limit:
        raise ValueError(
            f"Text length ({len(text)} chars) exceeds the {model} limit of {limit} characters."
        )


# ============================================================
# Audio Conversion
# ============================================================


def _format_extension(fmt: str) -> str:
    if fmt.startswith("mp3"):
        return ".mp3"
    if fmt.startswith("wav"):
        return ".wav"
    if fmt.startswith("opus"):
        return ".opus"
    if fmt.startswith("pcm"):
        return ".raw"
    return ".bin"


def silence_audio(seconds: float = 1.0, sample_rate: int = 44100, channels: int = 1) -> dict:
    """Return a ComfyUI AUDIO dict containing silence.

    Used as a non-crashing fallback when upstream API returns empty audio.
    1 second of silence at 44.1k mono is small enough that downstream
    spectrogram / resampler nodes won't crash on window-size assertions.
    """
    n = max(1, int(seconds * sample_rate))
    return {
        "waveform": torch.zeros(1, max(1, channels), n, dtype=torch.float32),
        "sample_rate": int(sample_rate),
    }


def audio_bytes_to_comfy(audio_bytes: bytes, output_format: str) -> dict:
    """Convert ElevenLabs API audio bytes to ComfyUI AUDIO dict.

    AUDIO dict shape: ``{"waveform": tensor[1, channels, samples], "sample_rate": int}``

    Handles:
    - Empty / truncated bytes — returns 1s of silence rather than crashing.
    - PCM with odd byte count — pads with one zero byte so int16 frombuffer
      doesn't ValueError on a truncated download.
    - Mono and stereo audio via soundfile / torchaudio fallback chain.
    - Uses BytesIO (no NamedTemporaryFile) to avoid Windows tempfile
      permission/cleanup issues.
    """
    if not audio_bytes:
        # Empty body — return silence and let the caller log; better than
        # blowing up downstream nodes that expect a non-empty tensor.
        return silence_audio(1.0)

    if output_format.startswith("pcm_"):
        try:
            sample_rate = int(output_format.split("_")[1])
        except (IndexError, ValueError):
            sample_rate = 44100
        # ElevenLabs PCM endpoints return signed 16-bit little-endian PCM,
        # mono. Defend against truncated stream (odd byte count).
        if len(audio_bytes) % 2:
            audio_bytes = audio_bytes + b"\x00"
        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            return silence_audio(1.0, sample_rate)
        waveform = torch.from_numpy(audio_np.copy()).unsqueeze(0).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": int(sample_rate)}

    # For WAV / MP3 / Opus — try soundfile (in-memory), fall back to torchaudio.
    try:
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(buf, dtype="float32", always_2d=False)
        if data.ndim == 1:
            waveform = torch.from_numpy(data.copy()).unsqueeze(0).unsqueeze(0)
        else:
            # soundfile returns (samples, channels) — transpose to (channels, samples)
            waveform = torch.from_numpy(data.T.copy()).unsqueeze(0)
        return {"waveform": waveform.float(), "sample_rate": int(sample_rate)}
    except Exception:
        pass

    try:
        import torchaudio
        buf = io.BytesIO(audio_bytes)
        # `format=` hint helps when there's no magic-bytes match for opus
        ext = _format_extension(output_format).lstrip(".")
        try:
            waveform, sample_rate = torchaudio.load(buf, format=ext)
        except Exception:
            buf.seek(0)
            waveform, sample_rate = torchaudio.load(buf)
        waveform = waveform.unsqueeze(0).float()
        return {"waveform": waveform, "sample_rate": int(sample_rate)}
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode {output_format} audio ({len(audio_bytes)} bytes). "
            f"Install soundfile or torchaudio. Error: {e}"
        )


def comfy_audio_to_bytes(audio: dict) -> tuple:
    """Convert ComfyUI AUDIO dict to WAV bytes for upload.

    Returns ``(bytes, sample_rate)``. Accepts AUDIO dicts with waveform
    shapes [samples], [channels, samples], or [batch, channels, samples]
    so we never crash on a downstream that hands us a stripped tensor.
    """
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(
            "AUDIO input must be a dict with 'waveform' tensor and 'sample_rate' int. "
            f"Got: {type(audio).__name__}"
        )

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if not isinstance(waveform, torch.Tensor):
        raise ValueError(
            f"AUDIO 'waveform' must be a torch.Tensor. Got: {type(waveform).__name__}"
        )

    # Normalize to (channels, samples) regardless of input shape
    if waveform.dim() == 3:
        waveform = waveform[0]               # [B,C,T] -> [C,T] (take first batch)
    if waveform.dim() == 2:
        # [C, T] — already in the right shape
        pass
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)     # [T] -> [1, T]
    else:
        raise ValueError(f"Unsupported waveform dim: {waveform.dim()} (shape {tuple(waveform.shape)})")

    # soundfile expects (samples, channels). Convert to float32 to avoid dtype surprises.
    data = waveform.detach().cpu().float().numpy().T   # -> (samples, channels)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    try:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), sample_rate
    except Exception:
        pass

    # Fallback manual WAV encoding (no soundfile available)
    return _manual_wav_encode(data, sample_rate), sample_rate


def _manual_wav_encode(samples_x_channels: np.ndarray, sample_rate: int) -> bytes:
    """Encode a (samples, channels) float32 array to 16-bit PCM WAV bytes."""
    if samples_x_channels.ndim == 1:
        samples_x_channels = samples_x_channels[:, np.newaxis]
    samples, channels = samples_x_channels.shape

    int_data = np.clip(samples_x_channels * 32767.0, -32768.0, 32767.0).astype(np.int16)
    # Interleave: int_data is (samples, channels); to bytes wants interleaved
    raw_bytes = int_data.reshape(-1).tobytes()

    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    data_size = len(raw_bytes)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw_bytes)
    return buf.getvalue()


# ============================================================
# Account-info PII redaction
# ============================================================

_PII_KEYS_REDACT = {
    "xi_api_key", "api_key", "first_name", "last_name", "email",
    "stripe_customer_id", "subscription_id", "phone_number",
    "billing_address", "billing_email", "user_id",
}


def redact_account_pii(data: dict) -> dict:
    """Return a deep copy of an /v1/user response with PII fields scrubbed.

    Removes name / email / billing / IDs but preserves subscription tier,
    usage counts, voice slots, and capability flags so AccountInfo nodes
    can still drive logic without leaking the user's identity into a
    workflow PNG saved to Civitai / Discord.
    """
    if not isinstance(data, dict):
        return data
    out = {}
    for k, v in data.items():
        if k in _PII_KEYS_REDACT:
            continue
        if isinstance(v, dict):
            out[k] = redact_account_pii(v)
        elif isinstance(v, list):
            out[k] = [redact_account_pii(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out
