"""Image, audio, and tensor conversion utilities for ComfyUI nodes."""

import base64
import io
import struct
from typing import Optional

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_base64(tensor: Optional[torch.Tensor], fmt: str = "PNG") -> Optional[str]:
    if tensor is None:
        return None
    img = tensor_to_pil(tensor)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_jpeg_bytes(tensor: torch.Tensor, quality: int = 95) -> bytes:
    img = tensor_to_pil(tensor)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(data))
    return pil_to_tensor(img)


def audio_to_comfy(audio_bytes: bytes, output_format: str = "wav") -> dict:
    try:
        import soundfile as sf
        buf = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buf, dtype="float32")
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]
        waveform = torch.from_numpy(audio_data.T).unsqueeze(0).float()
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        pass

    try:
        import torchaudio
        buf = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buf, format=output_format)
        waveform = waveform.unsqueeze(0).float()
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode audio. Install soundfile or torchaudio. Error: {e}"
        )


def comfy_to_audio_bytes(audio: dict, fmt: str = "wav") -> bytes:
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform[0]

    try:
        import soundfile as sf
        buf = io.BytesIO()
        data = waveform.cpu().numpy().T
        sf.write(buf, data, sample_rate, format="WAV")
        return buf.getvalue()
    except Exception:
        pass

    data = waveform.cpu().numpy()
    channels = data.shape[0]
    samples = data.shape[1]
    int_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    interleaved = np.empty(channels * samples, dtype=np.int16)
    for ch in range(channels):
        interleaved[ch::channels] = int_data[ch]

    raw_bytes = interleaved.tobytes()
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2

    buf = io.BytesIO()
    data_size = len(raw_bytes)
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
