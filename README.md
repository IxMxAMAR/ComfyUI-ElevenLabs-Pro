# ComfyUI-ElevenLabs-Pro

Full-featured ElevenLabs API integration for ComfyUI. **30 nodes**: TTS,
Speech-to-Speech, Sound Effects, Music, Voice Cloning / Design / Remix,
Multi-speaker Dialogue, Speech-to-Text, Forced Alignment, Audio Isolation —
plus 11 pure-Python audio utility nodes (subtitle export, concat, normalize,
trim, channels, save, cost estimator, …).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/IxMxAMAR/ComfyUI-ElevenLabs-Pro.git
pip install -r ComfyUI-ElevenLabs-Pro/requirements.txt
```

Then restart ComfyUI.

## API key

Set the `ELEVENLABS_API_KEY` environment variable, or paste your key into
the `ElevenLabs Pro - API Key` node. Keys are scrubbed from any raised
error message so they never end up in ComfyUI logs or saved workflow
PNG metadata.

## Nodes (30 total)

### Voice — 7 nodes
| Node | Purpose |
|------|---------|
| `Voice Selector` | Pick a predefined voice OR pass a custom `voice_id`. |
| `Fetch Voices` | List your account's voices (premade / cloned / generated / professional) with search + gender filter. Cached 5 min. |
| `Get Voice By Name` | Look up a `voice_id` by name (exact / contains / starts-with). |
| `Voice Clone` | Instant voice clone from up to 8 audio samples. **`create=False` by default** — flip to True to actually upload (prevents accidental slot/credit burn on workflow re-queue). |
| `Voice Design` | Generate 3 voice previews from a text description (`eleven_multilingual_ttv_v2` or `eleven_ttv_v3`, optional reference audio, loudness, guidance, seed). Returns all 3 IDs; pick one with `preview_index`. Sample text under 100 characters is auto-generated. |
| `Voice Remix` | Remix an existing voice from a description of the changes. Same outputs as Voice Design. |
| `Voice Create` | Save a designed or remixed preview as a permanent library voice (description of 20-1000 characters required). Also gated by `create=True`. |

### TTS — 5 nodes
| Node | Purpose |
|------|---------|
| `Text to Speech` | Full TTS with every API parameter (stability, similarity, style, speed, speaker boost, language, normalization, seed, previous/next text, PVC-as-IVC, enable_logging, pronunciation dicts, Japanese language normalization). Auto-validates eleven_v3 normalization, turbo language requirement. |
| `TTS with Timestamps` | TTS + character-level alignment JSON, with voice settings, language, normalization and previous/next text. |
| `Text to Dialogue` | Multi-speaker (up to 10) `eleven_v3` / `eleven_v4` dialogue with previous/future text, PVC-as-IVC and pronunciation dictionaries. Total text length validated. |
| `Text to Dialogue with Timestamps` | Same inputs as Text to Dialogue, plus character-level alignment JSON and per-voice segments JSON. |
| `Voice Tag Inserter` | Pure-text helper — dropdown of v3 expression tags (`[whispers]`, `[laughs]`, `[excited]`, …) injected into your prompt. |

### Audio (API) — 5 nodes
| Node | Purpose |
|------|---------|
| `Speech to Speech` | Transform an AUDIO input to another voice. |
| `Sound Effects` | Text → SFX with duration (or model-chosen duration) / prompt_influence / loop / seed. Uses the `eleven_text_to_sound_v2` model id. |
| `Audio Isolation` | Remove background noise / isolate vocals. |
| `Speech to Text` | Transcribe AUDIO via `scribe_v2` or `scribe_v2_medical`, with diarization, speaker roles, multichannel, audio-event tagging, keyterms, no-verbatim, transcript editing (`edited_text` output), temperature, seed. |
| `Forced Alignment` | Align a known transcript to AUDIO and get word-level timings (feeds Subtitle Export). |

### Music — 2 nodes
| Node | Purpose |
|------|---------|
| `Music Generation` | Text-prompt OR `composition_plan` JSON, with `music_v1` / `music_v2` / `music_v2_5`. 3 – 600 s. Lyrics, generation mode, finetunes, force instrumental, respect section durations, C2PA signing. |
| `Music Composition Plan` | Prompt → composition plan JSON to edit and feed into Music Generation. |

### Audio (pure-Python utilities) — 7 nodes
| Node | Purpose |
|------|---------|
| `Subtitle Export (SRT/VTT)` | Convert `TTSTimestamps` alignment or `STT` words into SRT or VTT. Cue grouping by char count + seconds. |
| `Audio Concat` | Stitch up to 8 clips. Resamples mismatched SRs (torchaudio or linear-interp fallback). Broadcasts mono → stereo. Optional silence gap. |
| `Audio Normalize` | Peak / RMS normalization to a target dBFS. |
| `Audio Metadata` | Read `duration_seconds`, `sample_rate`, `channels`, `samples`. |
| `Audio Channels` | to-mono / to-stereo / left-only / right-only / swap. |
| `Audio Trim` | Trim by start / end seconds. |
| `Save Audio` | Write AUDIO to disk under `ComfyUI/output` as wav / mp3 / flac / ogg. Sanitized filename + subdir (no `..` traversal). |

### Utilities — 2 nodes
| Node | Purpose |
|------|---------|
| `Cost Estimator` | Estimate credit cost from text + model **before** the API call. |
| `Voice Settings Preset` | Build a `voice_settings` JSON blob for reuse across workflows. |

### Config — 2 nodes
| Node | Purpose |
|------|---------|
| `API Key` | Hold the key and pass it to other nodes. `password=True` masks it in the UI. |
| `Account Info` | Subscription tier + character usage. **PII-redacted by default** — email, name, billing IDs are stripped before being returned so saved workflows don't leak your identity. |

## Models

### TTS
- `eleven_v4` — most expressive, 90+ languages (10 000 char limit)
- `eleven_v3` — expressive, supports `[whispers]`/`[laughs]`/v3 tags (5 000 char limit, auto-forces normalization=off)
- `eleven_multilingual_v2` (10 000 chars)
- `eleven_flash_v2_5`, `eleven_turbo_v2_5` (40 000 chars, **require explicit language**)
- `eleven_flash_v2`, `eleven_turbo_v2` (English-only, 30 000 chars)

`eleven_monolingual_v1` and `eleven_multilingual_v1` were removed by ElevenLabs on 2026-07-09.

### Dialogue
- `eleven_v3`, `eleven_v4`

### Speech-to-Speech
- `eleven_multilingual_sts_v2`, `eleven_english_sts_v2`

### Speech-to-Text
- `scribe_v2`, `scribe_v2_medical` (`scribe_v1` was removed on 2026-07-09)

### Voice Design
- `eleven_multilingual_ttv_v2`, `eleven_ttv_v3`

### SFX
- `eleven_text_to_sound_v2`

### Music
- `music_v1`, `music_v2`, `music_v2_5`

## Output formats

All standard ElevenLabs formats supported:
mp3 (44.1k @ 192/128/96/64/32, 24k @ 48, 22k @ 32),
wav (44.1k / 48k / 32k / 24k / 22k / 16k / 8k),
pcm (44.1k / 48k / 32k / 24k / 22k / 16k / 8k),
opus (48k @ 192/128/96/64/32).
Music Generation also accepts mp3 (48k @ 128/192/240/320).

The PCM path is decoded directly as int16 little-endian mono so it works
without an MP3 decoder. WAV / MP3 / Opus use `soundfile` first, fall
back to `torchaudio`.

## Security

This v2.1 release went through a top-to-bottom audit. Highlights:

- **API key never leaks.** `scrub_secrets()` removes the key from every
  raised exception, error message, and the IS_CHANGED hash (so the key
  never enters workflow PNG metadata).
- **AccountInfo PII redacted.** Email, name, billing IDs stripped
  before returning. Uses `/v1/user/subscription` by default (no PII).
- **VoiceClone / VoiceCreate require explicit `create=True`.** v2.0
  burned a voice slot on every queue prompt; v2.1 default is dry-run.
- **All paid API nodes cache by input hash.** v2.0 used
  `IS_CHANGED = NaN` which forced every queue to re-bill the API.
- **Download size cap (500 MB default).** No more OOM from a malicious
  endpoint.
- **Multipart uploads survive retries.** File pointers reset to 0
  between attempts (v2.0 sent empty bodies on retry).
- **SaveAudio rejects path traversal.** Subdirectory and filename are
  sanitized so a malicious workflow can't write `../../etc/passwd`.

## Tests

```bash
cd ComfyUI/custom_nodes/ComfyUI-ElevenLabs-Pro
python -m pytest tests/ -c pytest.ini
```

**153 / 153 passing.** All HTTP mocked — no real ElevenLabs credits
consumed during testing.

## License

MIT.
