"""
ComfyUI-ElevenLabs-Pro
Full-featured ElevenLabs integration for ComfyUI.
Exposes ALL API parameters, models, voices, languages, and output formats.
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Fallback for pytest / standalone (no parent package context)
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
