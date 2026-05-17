"""Pytest config — make the package importable from a parent path.

ComfyUI-ElevenLabs-Pro has a dash in the name so we can't `import
comfyui_elevenlabs_pro` directly. Inject the repo root into sys.path so
the `shared` and `utils` modules resolve.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
