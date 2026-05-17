"""Shared node utilities and mixins for ComfyUI nodes."""

import hashlib
import json


class AlwaysExecuteMixin:
    """Mixin that forces re-execution on every queue (no stale cache).

    Use SPARINGLY — for paid API nodes, prefer InputCacheMixin so identical
    inputs are cached and don't burn credits on every queue.
    """

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class InputCacheMixin:
    """Cache node execution by input hash so identical inputs don't re-run.

    Critical for paid API nodes (TTS, STS, Music, SFX, STT, AudioIsolation,
    VoiceClone, VoiceDesign): without this, ComfyUI re-executes on every
    queue prompt, burning credits.

    Audio tensors are not hashable by value cheaply — we hash shape+sample_rate
    instead, which means re-running with a different tensor of the same shape
    will still be re-cached. Acceptable trade-off for audio inputs.
    """

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _hash_inputs(kwargs)


def _hash_inputs(kwargs: dict) -> str:
    """Stable hash of kwargs, with audio/tensor inputs hashed by shape+rate."""
    parts = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        # Skip the api_key entirely from the hash — never include secrets
        # in IS_CHANGED return values (they propagate to workflow PNG metadata).
        if k == "api_key" or k.endswith("_key") or k.endswith("api_key"):
            continue
        if isinstance(v, dict) and "waveform" in v and "sample_rate" in v:
            # AUDIO dict — hash shape + sample_rate + a few sample values
            try:
                wf = v["waveform"]
                sr = v["sample_rate"]
                # Tuple of shape + sr + first/last/mean value for content fingerprint
                shape = tuple(wf.shape)
                fp = (
                    shape, int(sr),
                    float(wf.flatten()[0].item()) if wf.numel() else 0.0,
                    float(wf.flatten()[-1].item()) if wf.numel() else 0.0,
                    float(wf.float().mean().item()) if wf.numel() else 0.0,
                )
                parts.append(f"{k}=audio:{fp}")
            except Exception:
                parts.append(f"{k}=audio:opaque")
        elif hasattr(v, "shape") and hasattr(v, "numel"):
            # Plain tensor
            try:
                fp = (tuple(v.shape), float(v.float().mean().item()) if v.numel() else 0.0)
                parts.append(f"{k}=tensor:{fp}")
            except Exception:
                parts.append(f"{k}=tensor:opaque")
        else:
            try:
                parts.append(f"{k}={json.dumps(v, sort_keys=True, default=str)}")
            except Exception:
                parts.append(f"{k}={type(v).__name__}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def redact_secrets(text: str, secrets: list) -> str:
    """Redact any of `secrets` (non-empty strings) from `text`.

    Use before logging error bodies / printing — paranoid defense in case
    a key ended up in a response body or exception message.
    """
    if not text:
        return text
    out = text
    for s in secrets:
        if s and isinstance(s, str) and len(s) >= 8:
            out = out.replace(s, "***REDACTED***")
    return out
