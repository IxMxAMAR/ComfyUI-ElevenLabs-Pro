# ComfyUI-ElevenLabs-Pro

15 nodes for the full ElevenLabs API. TTS that actually sounds like a person said it, voice cloning, music generation, sound effects, and more.

Part of [ComfyUI-API-Toolkit](https://github.com/IxMxAMAR/ComfyUI-API-Toolkit) — also available as a standalone package because not everyone needs 60 nodes at once.

---

## Installation

**ComfyUI Manager** (easiest): Search "ElevenLabs Pro" and hit install.

**Registry:**
```
comfy node registry-install comfyui-elevenlabs-pro
```

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/IxMxAMAR/ComfyUI-ElevenLabs-Pro
pip install requests soundfile scipy
```

Then restart ComfyUI.

---

## Getting an API Key

Go to [elevenlabs.io](https://elevenlabs.io) -> Profile -> API Keys -> generate one. The free tier is generous enough to figure out if you like it.

Add it to a workflow via the **ElevenLabs API Key** node, or set `ELEVENLABS_API_KEY` in your environment if you don't want it floating around in workflow files.

---

## Nodes (15 total)

### Config

**ElevenLabs API Key**
Password-masked input. Supports env var fallback so you're not hardcoding secrets into workflows you share. Feeds into every other node.

**ElevenLabs Account Info**
Check your remaining credits and subscription tier. Useful before you kick off a big batch job and find out mid-run you're out of credits.

---

### Voice

**ElevenLabs Voice Selector**
Dropdown of 41 built-in voices. No API call needed. Good starting point before you go down the voice cloning rabbit hole.

**ElevenLabs Fetch Voices**
Fetches your full voice library with pagination. Results are cached for 5 minutes so you're not hammering the API every time you re-queue. Has a `force_refresh` toggle for when you've just cloned a voice and it hasn't shown up yet.

**ElevenLabs Voice Clone**
Clone a voice from audio samples — up to 8 files. The more samples you provide, the better it sounds. Name it, add a description, and it goes straight into your ElevenLabs library.

**ElevenLabs Voice Design**
Describe a voice in plain text and get one back. "Warm, mid-30s American woman, slight rasp" — that kind of thing. Returns a preview so you can iterate before committing.

**ElevenLabs Voice Create**
Takes the output of Voice Design and saves it as a permanent voice in your library. Voice Design gives you a temp preview; this makes it stick.

---

### TTS

**ElevenLabs Text to Speech**
The main event. Full parameter control: stability, similarity boost, style exaggeration, speed, speaker boost. 27 output formats including MP3 at various bitrates, WAV, PCM, and Opus. Supports all current models: `eleven_v3`, `eleven_multilingual_v2`, `eleven_flash_v2_5`, `eleven_flash_v2`.

Character limit validation runs before the API call, which saves you from the extremely unhelpful 422 error you'd otherwise get.

**ElevenLabs TTS with Timestamps**
Same as Text to Speech, but also outputs word-level timing data as JSON. Useful if you're syncing audio to video, building captions, or doing anything that needs to know exactly when each word hits.

**ElevenLabs Text to Dialogue**
Multi-speaker TTS. Up to 10 speakers, each with their own voice and text. Feed it a list of lines and get back a single audio file with all of them stitched together. Good for explainer content, fake interviews, and audiobook-style narration.

---

### Audio

**ElevenLabs Speech to Speech**
Transform one voice into another. Feed it audio and a target voice, and it re-voices it while preserving the original delivery. Useful when you have a scratch recording with good performance but a voice that doesn't fit.

**ElevenLabs Sound Effects**
Text-to-SFX. Describe a sound, get audio back. "Heavy wooden door creaking open slowly" works. Duration and prompt influence are configurable.

**ElevenLabs Audio Isolation**
Strips background noise and extracts the vocal track. Handy for cleaning up reference audio before voice cloning.

**ElevenLabs Speech to Text**
Transcription via Scribe v2, which supports speaker diarization — so it can tell you not just what was said but who said it (or at least, who said it relative to the other speakers). Returns the transcript as a string.

---

### Music

**ElevenLabs Music Generation**
Text-to-music. Describe a vibe, get a track. Instrumental toggle if you want it without vocals. It's not Suno, but it's API-accessible and fits in a workflow, which is the whole point.

---

## Features Worth Knowing About

- **Retry with exponential backoff** on all API calls. Transient failures don't blow up your workflow.
- **Voice cache** (5 min) so Fetch Voices isn't hitting the API on every run.
- **Character limit validation** before TTS calls — fails fast with a clear message instead of a cryptic HTTP error.
- **IS_CHANGED** on all generative nodes so ComfyUI re-runs them when inputs change rather than serving stale cache.
- **Tooltips on everything.** Hover over any input if you're not sure what it does.
- **Structured error messages.** Not raw JSON dumps.
- **Password-masked API key field.** Because shoulder surfing is real.

---

## Models

| Model | Notes |
|---|---|
| `eleven_v3` | Best quality, highest latency |
| `eleven_multilingual_v2` | 29 languages, solid quality |
| `eleven_flash_v2_5` | Fast, good enough for most use cases |
| `eleven_flash_v2` | Fastest, lowest latency |

---

## Requirements

- ComfyUI
- Python 3.10+
- `requests`, `soundfile`, `scipy`
- An ElevenLabs account (free tier works fine to get started)

---

## Credits

Made by [IxMxAMAR](https://github.com/IxMxAMAR).

If something's broken or you want a node that doesn't exist yet, open an issue.
