"""
ElevenLabs API client -- constants, voice cache, API helpers, audio conversion.

All HTTP calls go through shared.retry.api_request_with_retry for automatic
exponential-backoff on transient errors (429 / 5xx).
"""

import io
import os
import time
import hashlib
import tempfile

import numpy as np
import torch

from .shared.retry import api_request_with_retry
from .shared.errors import APIPermanentError

# ============================================================
# Constants
# ============================================================

ELEVENLABS_API_BASE = "https://api.elevenlabs.io"

TTS_MODELS = [
    "eleven_v3",
    "eleven_multilingual_v2",
    "eleven_flash_v2_5",
    "eleven_flash_v2",
]

STS_MODELS = [
    "eleven_multilingual_sts_v2",
    "eleven_english_sts_v2",
]

STT_MODELS = [
    "scribe_v2",
]

SFX_MODELS = [
    "eleven_sfx_v2",
]

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
    "eleven_flash_v2": 40000,
}

# ============================================================
# Voice Cache
# ============================================================

_voice_cache: dict = {}
_voice_cache_time: dict = {}
_VOICE_CACHE_TTL = 300  # 5 minutes

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


def api_post(url: str, api_key: str, timeout: float = 300, **kwargs) -> "requests.Response":
    """POST with retry logic.  Accepts the same kwargs as requests (json, data, files, params, headers)."""
    headers = kwargs.pop("headers", None) or {"xi-api-key": api_key}
    if "Content-Type" not in headers and "files" not in kwargs and "data" not in kwargs:
        headers["Content-Type"] = "application/json"
    return api_request_with_retry(
        "POST",
        url,
        headers=headers,
        timeout=timeout,
        service_name="ElevenLabs",
        **kwargs,
    )


def api_get(url: str, api_key: str, timeout: float = 30, **kwargs) -> "requests.Response":
    """GET with retry logic."""
    headers = kwargs.pop("headers", None) or {"xi-api-key": api_key}
    return api_request_with_retry(
        "GET",
        url,
        headers=headers,
        timeout=timeout,
        service_name="ElevenLabs",
        **kwargs,
    )


def check_response(resp) -> None:
    """Raise APIPermanentError with parsed message if status >= 400.

    Normally api_post / api_get already raise through the retry layer,
    but this helper is useful for multipart/file uploads that bypass
    the standard JSON path.
    """
    if resp.status_code >= 400:
        from .shared.errors import parse_error_response
        raise parse_error_response("ElevenLabs", resp.status_code, resp.text)


def fetch_all_voices(api_key: str, force_refresh: bool = False) -> list:
    """Paginated v2 voices with 5-min cache."""
    cache_key = hashlib.md5(api_key.encode()).hexdigest()[:8]
    now = time.time()

    if (
        not force_refresh
        and cache_key in _voice_cache
        and (now - _voice_cache_time.get(cache_key, 0)) < _VOICE_CACHE_TTL
    ):
        return _voice_cache[cache_key]

    voices = []
    next_token = None

    while True:
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
            labels = v.get("labels", {})
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

    _voice_cache[cache_key] = voices
    _voice_cache_time[cache_key] = now
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


def audio_bytes_to_comfy(audio_bytes: bytes, output_format: str) -> dict:
    """Convert ElevenLabs API audio bytes to ComfyUI AUDIO dict."""
    if output_format.startswith("pcm_"):
        sample_rate = int(output_format.split("_")[1])
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

    # For WAV, MP3, Opus -- use soundfile or torchaudio
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=_format_extension(output_format), delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            data, sample_rate = sf.read(tmp_path, dtype="float32")
        finally:
            os.unlink(tmp_path)

        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        pass

    # Fallback: torchaudio
    import torchaudio
    buf = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buf)
    waveform = waveform.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def comfy_audio_to_bytes(audio: dict) -> tuple:
    """Convert ComfyUI AUDIO dict to WAV bytes for upload. Returns (bytes, sample_rate)."""
    import soundfile as sf

    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 2:
        waveform = waveform.T  # [samples, channels]

    audio_np = waveform.cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    buf.seek(0)
    return buf.getvalue(), sample_rate
