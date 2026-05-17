# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] — 2026-05-17

Re-publish of v2.1.0 with corrected registry version number (2.1.x was
reserved by a prior partial publish on the registry side).

## [2.1.0] — 2026-05-17

Big audit + fix release driven by two full-codebase Gemini Pro reviews
(API client / audio I/O + ComfyUI node UX) plus first-party verification
of every Critical/High finding against the live ElevenLabs API spec.
**3 credit-burning bugs, 5 API contract bugs, 4 security/leak fixes, 11
new utility nodes, 101 new tests.**

### Security — fix immediately if you ever share workflows

- **API keys scrubbed from error messages.** `parse_error_response` and
  the new `scrub_secrets()` helper redact `xi-api-key=…` URL params,
  `?api_key=…` patterns, and any explicit `extra_secrets` token before
  the message reaches ComfyUI's UI / console / saved workflow PNGs.
  Previously, a `requests` `ConnectionError` could echo
  `https://api.elevenlabs.io/v1/?xi-api-key=SECRET` into a stack trace.
- **AccountInfo PII redacted by default.** `/v1/user` returns email,
  first/last name, billing IDs, Stripe customer ID. v2.0 returned the
  raw JSON straight back — anyone who saved a workflow PNG with that
  node and shared it on Civitai/Discord leaked their identity. New
  `redact_account_pii()` deep-strips PII fields and now defaults to
  querying `/v1/user/subscription` instead (cleaner response, no PII).
- **Download size cap (DoS guard).** `download_file()` previously read
  chunks into an unbounded `io.BytesIO`. A malicious / misconfigured
  endpoint could exhaust RAM. New `DEFAULT_MAX_DOWNLOAD_BYTES=500 MB`
  cap honors `Content-Length` upfront AND streams-and-aborts.
- **Multipart retry safety.** `requests` does not rewind file-like
  bodies on retry — a 503 on attempt 1 would re-POST an empty body on
  attempt 2 and 400. New `_seekable_files()` resets all
  `BytesIO`/file-like payloads in `files=` between attempts.

### Fixed — Credit-burning bugs (HIGHEST USER IMPACT)

- **`IS_CHANGED = NaN` removed from paid nodes.** v2.0 used
  `AlwaysExecuteMixin` (returns `float("nan")`, which `!=` itself, so
  every queue prompt re-runs the node) on TTS, STS, SFX, Music, STT,
  AudioIsolation, VoiceClone, VoiceDesign, VoiceCreate, Dialogue,
  TTSTimestamps. **Every workflow re-queue re-billed the user.** New
  `InputCacheMixin` hashes (text + voice + settings + audio shape) so
  identical inputs hit ComfyUI's native cache. API key is explicitly
  excluded from the hash (so swapping keys doesn't invalidate, and the
  key never enters workflow PNG metadata via the IS_CHANGED return).
  `FetchVoices` and `AccountInfo` keep `AlwaysExecute` (no API cost
  beyond a cached GET).
- **`VoiceClone` requires explicit `create=True`.** v2.0 created a new
  voice every queue — within minutes a user could blow their voice slot
  cap. v2.1 default is `create=False`, returns a dry-run status string;
  set `True` to actually upload.
- **`VoiceCreate` requires explicit `create=True`.** Same fix; v2.0
  saved a new preview to library on every queue.

### Fixed — API contract bugs (silent failures in v2.0)

- **`Music` switched to `music_length_ms`.** v2.0 sent
  `duration_seconds` (not a valid field for `/v1/music`) and capped at
  300s. Verified against current ElevenLabs spec: field is
  `music_length_ms` (range 3 000 – 600 000), max duration 10 min.
  Endpoint now also exposes `model_id=music_v1`, `composition_plan`,
  `respect_sections_durations`, `store_for_inpainting`, `sign_with_c2pa`.
- **`SFX` default `model_id` corrected.** v2.0 sent `eleven_sfx_v2`
  (wrong; not accepted by the live API). Correct id per docs is
  `eleven_text_to_sound_v2`; `eleven_sfx_v2` is kept in the dropdown as
  a fallback for the legacy spelling.
- **`TTS` for `eleven_v3` forces `apply_text_normalization="off"`.** v3
  is the only allowed value per ElevenLabs spec; v2.0 sent `"auto"` and
  the API silently degraded (or 422'd). New `_enforce_v3_normalization()`
  auto-downgrades with a console note.
- **`TTS` `eleven_turbo_v2_5` / `eleven_flash_v2_5` require explicit
  language.** v2.0 let users pick "Auto Detect" and the API 422'd. Now
  raised before the request: `Model X requires an explicit language code`.
- **`STT` `timestamps_granularity="none"` actually sends `none`.** v2.0
  silently omitted the field when the user picked `"none"` (so the API
  returned word-level timestamps anyway). The API accepts `"none"` per
  spec — verified.
- **`TTS` malformed `pronunciation_dictionary_locators` JSON is now an
  error, not silently swallowed.** v2.0 had `except json.JSONDecodeError:
  pass` — user got no feedback their dict didn't apply.

### Fixed — Audio I/O correctness

- **PCM decode survives truncated streams.** v2.0's
  `np.frombuffer(b, dtype=np.int16)` raised `ValueError` on odd byte
  counts. v2.1 pads with `b"\x00"` so a clipped chunked-encoding
  download degrades gracefully.
- **Empty audio bytes return silence, not crash.** Both `audio_to_comfy`
  and the PCM path now return `silence_audio(1.0)` (1 second of mono
  silence at 44.1 kHz) when the API returns an empty body, instead of
  bubbling up a `RuntimeError` to the user.
- **Fallback silence is 1 s, not 1 sample.** `torch.zeros(1, 1, 1)` was
  a syntactically-valid AUDIO dict but crashed downstream STFT/
  spectrogram nodes that require ≥ window-size samples.
- **`comfy_audio_to_bytes` accepts 1-D / 2-D / 3-D waveforms.** v2.0
  required `[B, C, T]`; an upstream that handed us `[T]` (mono only)
  crashed.
- **NamedTemporaryFile replaced with `BytesIO`.** Windows refused to
  reopen the tempfile with `soundfile` while the handle was still alive;
  `delete=False` also leaked the tempfile on any exception. In-memory
  buffer eliminates both classes of bug.
- **`always_2d=False` on `soundfile.read`.** Avoids spurious channel
  expansion on stereo files.
- **`parse_error_response` survives binary response bodies.** v2.0
  passed `response.text` straight to JSON parse; a 500 from the binary
  TTS endpoint produced a UnicodeDecodeError or megabyte-of-garbage
  detail string. v2.1 sniffs `Content-Type` and returns
  `<binary response>` for non-text bodies.

### Fixed — Retry / backoff hardening

- **`Retry-After: <HTTP-date>` now respected.** v2.0 only parsed
  `Retry-After: 120` (seconds). RFC 7231 also allows
  `Retry-After: Fri, 31 Dec 2026 23:59:59 GMT` — now handled.
- **Exponential backoff clamped + jittered.** Was `2 * 2^attempt`
  unbounded. v2.1 caps at `max_delay=60s` and adds ±25% jitter to
  prevent retry storms from concurrent ComfyUI workers.
- **Voice cache bounded LRU.** v2.0 stored one entry per distinct
  api-key hash, unbounded — long-running ComfyUI servers leaked. v2.1
  cap at 32 entries via `OrderedDict.popitem(last=False)`.
- **Pagination hard-cap.** Belt-and-braces: in addition to the existing
  `has_more=True && next_page_token=None` defensive break, also cap at
  100 pages so a misbehaving API can never wedge the worker.

### Added — 11 new utility nodes

- **`GetVoiceByName`** — look up a `voice_id` by name (exact / contains
  / starts-with). Convenience over `FetchVoices` + grep.
- **`VoiceTagInserter`** — dropdown of v3 expression tags (`[whispers]`,
  `[laughs]`, `[excited]`, …) injected into your text. Pure-Python.
- **`SubtitleExport`** — turn `TTSTimestamps` alignment OR `STT`
  `words_json` into SRT or VTT subtitles. Cue-grouping by char count
  and seconds.
- **`AudioConcat`** — stitch up to 8 AUDIO inputs along the time axis
  with optional silence gap. Resamples mismatched SRs (torchaudio if
  available, linear interp fallback). Broadcasts mono → stereo.
- **`AudioNormalize`** — peak / RMS normalization to a target dBFS.
- **`AudioMetadata`** — read `duration_seconds`, `sample_rate`,
  `channels`, `samples` out of any AUDIO dict.
- **`AudioChannels`** — to-mono / to-stereo / left-only / right-only /
  swap-channels.
- **`AudioTrim`** — clip AUDIO by start / end seconds.
- **`CostEstimator`** — estimate credit cost for a TTS call *before*
  making it (per-char × model multiplier).
- **`VoiceSettingsPreset`** — build a `voice_settings` JSON string for
  reuse across workflows (consistent character voices).
- **`SaveAudio`** — write AUDIO to disk under ComfyUI/output with format
  picker (wav / mp3 / flac / ogg). Subdirectory + filename are
  sanitized against `..` traversal.

### Added — Quality-of-life

- **`enable_logging` toggle** on TTS / TTSTimestamps / STS / Dialogue.
  Set to `False` for HIPAA / zero-retention requests (ElevenLabs does
  not store the audio or text).
- **`FetchVoices` gender filter** added; output `count` int.
- **`VoiceDesign` exposes all 3 previews.** v2.0 threw away the 2 you
  paid for. v2.1 adds `preview_index` (0..2) and an `all_voice_ids`
  output containing all returned `generated_voice_id`s.
- **`Dialogue` validates total text length** against the model's char
  limit *before* the API call. Adds explicit `model` dropdown (still
  `eleven_v3` for now; future-proof). Default `apply_text_normalization`
  is `"off"` (required for v3).
- **`STT` adds `scribe_v1`** to the model dropdown.
- **`SFX` validates duration** against the live 0.5 – 30 s window.
- **`previous_text` / `next_text` tooltips** clarified — they are
  context only, do NOT generate audio for that text.
- **`api_post` / `api_get` scrub the key from any raised exception.**
  Defense in depth against transport-layer leaks.

### Tests

- **101 / 101 passing** (was 0 — no test suite existed).
- `tests/test_audio_conversions.py` (13) — AUDIO dict shape, PCM
  round-trip, empty / truncated handling, 1-D / 2-D / 3-D waveform
  acceptance, silence fallback.
- `tests/test_errors_scrubbing.py` (15) — secret-leakage regression,
  API error parsing (JSON / non-JSON / binary), HTTP-code mapping.
- `tests/test_retry.py` (18) — retry on 5xx / connection errors, no
  retry on 4xx, file-pointer reset between attempts, `Retry-After`
  seconds + HTTP-date parsing, download size cap.
- `tests/test_pagination.py` (7) — voice pagination terminates on every
  weird response shape, LRU cache bounded.
- `tests/test_nodes.py` (48) — VoiceClone/Create safety toggles,
  AccountInfo PII redaction, TTS validation (empty text, missing voice,
  turbo language requirement, v3 normalization override, malformed
  pronunciation JSON), Music API contract (`music_length_ms` not
  `duration_seconds`, composition_plan, 600 s max), SFX model id and
  duration bounds, all 11 utility nodes, `InputCacheMixin` excludes
  api_key from hash, SaveAudio path-traversal rejection.

All tests mock HTTP via `unittest.mock` — **zero real ElevenLabs API
calls** during testing, no credits burned.

### Migration notes from v2.0

- `VoiceClone` and `VoiceCreate` now have a `create` boolean. Existing
  workflows that relied on auto-creation must flip it to `True`.
- `Music`'s `duration_seconds` is unchanged on the node UI but it's
  converted internally to `music_length_ms`. Workflows continue to work.
- `AccountInfo`'s `account_json` output is now redacted; if you had a
  downstream node parsing your email out of it, you need the raw
  endpoint (`endpoint="user"`) AND should know it leaks PII.
- `pronunciation_dictionary_locators` now raises on invalid JSON
  instead of silently dropping. Validate your JSON.
- New node IDs: `ElevenLabsPro_GetVoiceByName`,
  `ElevenLabsPro_VoiceTagInserter`, `ElevenLabsPro_SubtitleExport`,
  `ElevenLabsPro_AudioConcat`, `ElevenLabsPro_AudioNormalize`,
  `ElevenLabsPro_AudioMetadata`, `ElevenLabsPro_AudioChannels`,
  `ElevenLabsPro_AudioTrim`, `ElevenLabsPro_CostEstimator`,
  `ElevenLabsPro_VoiceSettingsPreset`, `ElevenLabsPro_SaveAudio`.

## [2.0.0] — 2026-04-12

- Initial public release. 15 nodes covering TTS, STS, SFX, Music,
  Voice Cloning, Voice Design, Voice Create, Speech-to-Text, Audio
  Isolation, Dialogue, plus API key + account info config nodes.
- Verified TTS/STS/STT model lists against ElevenLabs API.
- Voice pagination with 5-minute cache.
