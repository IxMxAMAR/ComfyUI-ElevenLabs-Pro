"""Audio conversion / AUDIO dict shape tests.

ComfyUI AUDIO contract: ``{"waveform": tensor[B, channels, samples], "sample_rate": int}``.
These tests guard against silent shape regressions.
"""
import io
import struct

import numpy as np
import pytest
import torch

from utils import (
    audio_bytes_to_comfy,
    comfy_audio_to_bytes,
    silence_audio,
)


def _make_wav_bytes(sample_rate=22050, channels=1, seconds=0.1, freq=440):
    """Construct a minimal valid WAV in memory using soundfile."""
    sf = pytest.importorskip("soundfile")
    n = int(sample_rate * seconds)
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels == 2:
        data = np.stack([signal, signal * 0.5], axis=-1)
    else:
        data = signal
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------- audio_bytes_to_comfy ---------

def test_audio_bytes_to_comfy_returns_audio_dict_shape():
    wav = _make_wav_bytes(sample_rate=22050, channels=1, seconds=0.1)
    out = audio_bytes_to_comfy(wav, "wav_22050")
    assert isinstance(out, dict)
    assert "waveform" in out
    assert "sample_rate" in out
    assert isinstance(out["waveform"], torch.Tensor)
    assert isinstance(out["sample_rate"], int)
    assert out["waveform"].dim() == 3, f"expected [B,C,T], got {out['waveform'].shape}"
    assert out["waveform"].shape[0] == 1  # batch
    assert out["waveform"].shape[1] == 1  # channels (mono)
    assert out["waveform"].dtype == torch.float32


def test_audio_bytes_to_comfy_stereo():
    wav = _make_wav_bytes(sample_rate=44100, channels=2, seconds=0.05)
    out = audio_bytes_to_comfy(wav, "wav_44100")
    assert out["waveform"].dim() == 3
    assert out["waveform"].shape[1] == 2  # stereo
    assert out["sample_rate"] == 44100


def test_audio_bytes_to_comfy_empty_returns_silence():
    """Empty bytes should not crash — return silence instead."""
    out = audio_bytes_to_comfy(b"", "mp3_44100_128")
    assert out["waveform"].dim() == 3
    assert out["waveform"].abs().sum().item() == 0.0


def test_audio_bytes_to_comfy_pcm_odd_byte_count_doesnt_crash():
    """Truncated PCM stream (odd byte count) must not crash np.frombuffer."""
    odd = b"\x00\x01\x02"  # 3 bytes, odd
    out = audio_bytes_to_comfy(odd, "pcm_44100")
    assert out["waveform"].dim() == 3
    assert out["sample_rate"] == 44100


def test_audio_bytes_to_comfy_pcm_round_trip():
    """Encode known PCM int16, decode, verify amplitude preserved."""
    samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    out = audio_bytes_to_comfy(samples.tobytes(), "pcm_22050")
    assert out["sample_rate"] == 22050
    wf = out["waveform"].squeeze().numpy()
    # 32767/32768 = 0.99996...
    assert abs(wf[1] - (16384 / 32768.0)) < 1e-6
    assert abs(wf[3] - (32767 / 32768.0)) < 1e-3


# --------- comfy_audio_to_bytes ---------

def test_comfy_audio_to_bytes_3d_tensor():
    """Standard ComfyUI AUDIO dict with [B,C,T] tensor."""
    wf = torch.zeros(1, 1, 22050, dtype=torch.float32)
    audio = {"waveform": wf, "sample_rate": 22050}
    out_bytes, sr = comfy_audio_to_bytes(audio)
    assert isinstance(out_bytes, bytes)
    assert len(out_bytes) > 44  # at least the WAV header
    assert sr == 22050
    assert out_bytes[:4] == b"RIFF"


def test_comfy_audio_to_bytes_2d_tensor():
    """[C,T] tensor (no batch dim) should still work."""
    wf = torch.zeros(2, 1000, dtype=torch.float32)
    audio = {"waveform": wf, "sample_rate": 16000}
    out, sr = comfy_audio_to_bytes(audio)
    assert sr == 16000
    assert len(out) > 44


def test_comfy_audio_to_bytes_1d_tensor():
    """[T] tensor (1D mono) should still work."""
    wf = torch.zeros(1000, dtype=torch.float32)
    audio = {"waveform": wf, "sample_rate": 8000}
    out, sr = comfy_audio_to_bytes(audio)
    assert sr == 8000
    assert len(out) > 44


def test_comfy_audio_to_bytes_invalid_input_raises():
    with pytest.raises(ValueError):
        comfy_audio_to_bytes({"foo": "bar"})


def test_comfy_audio_to_bytes_non_tensor_raises():
    with pytest.raises(ValueError):
        comfy_audio_to_bytes({"waveform": "not a tensor", "sample_rate": 22050})


def test_roundtrip_wav_audio_dict():
    """Round-trip an AUDIO dict through encode -> decode and verify shape."""
    wf = torch.randn(1, 1, 8000, dtype=torch.float32) * 0.1
    audio = {"waveform": wf, "sample_rate": 16000}
    raw, sr = comfy_audio_to_bytes(audio)
    decoded = audio_bytes_to_comfy(raw, "wav_16000")
    assert decoded["sample_rate"] == 16000
    assert decoded["waveform"].dim() == 3
    assert decoded["waveform"].shape[0] == 1
    assert decoded["waveform"].shape[1] == 1
    assert decoded["waveform"].shape[2] == 8000


# --------- silence_audio ---------

def test_silence_audio_shape_and_silence():
    s = silence_audio(0.5, sample_rate=44100, channels=2)
    assert s["sample_rate"] == 44100
    assert s["waveform"].dim() == 3
    assert s["waveform"].shape == (1, 2, 22050)
    assert s["waveform"].abs().sum().item() == 0.0


def test_silence_audio_minimum_samples():
    """Even with a tiny duration we should return at least 1 sample."""
    s = silence_audio(0.0)
    assert s["waveform"].shape[-1] >= 1
