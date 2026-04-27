"""Routing helpers for inbound user-attached images.

Two modes:

  native  — attach images as OpenAI-style ``image_url`` content parts on the
            user turn. Provider adapters (Anthropic, Gemini, Bedrock, Codex,
            OpenAI chat.completions) already translate these into their
            vendor-specific multimodal formats.

  text    — run ``vision_analyze`` on each image up-front and prepend the
            description to the user's text. The model never sees the pixels;
            it only sees a lossy text summary. This is the pre-existing
            behaviour and still the right choice for non-vision models.

The decision is made once per message turn by :func:`decide_image_input_mode`.
It reads ``agent.image_input_mode`` from config.yaml (``auto`` | ``native``
| ``text``, default ``auto``) and the active model's capability metadata.

In ``auto`` mode:
  - If the user has explicitly configured ``auxiliary.vision.provider``
    (i.e. not ``auto`` and not empty), we assume they want the text pipeline
    regardless of the main model — they've opted in to a specific vision
    backend for a reason (cost, quality, local-only, etc.).
  - Otherwise, if the active model reports ``supports_vision=True`` in its
    models.dev metadata, we attach natively.
  - Otherwise (non-vision model, no explicit override), we fall back to text.

This keeps ``vision_analyze`` surfaced as a tool in every session — skills
and agent flows that chain it (browser screenshots, deeper inspection of
URL-referenced images, style-gating loops) keep working. The routing only
affects *how user-attached images on the current turn* are presented to the
main model.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_VALID_MODES = frozenset({"auto", "native", "text"})


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes."""
    if not isinstance(raw, str):
        return "auto"
    val = raw.strip().lower()
    if val in _VALID_MODES:
        return val
    return "auto"


def _explicit_aux_vision_override(cfg: Optional[Dict[str, Any]]) -> bool:
    """True when the user configured a specific auxiliary vision backend.

    An explicit override means the user *wants* the text pipeline (they're
    paying for a dedicated vision model), so we don't silently bypass it.
    """
    if not isinstance(cfg, dict):
        return False
    aux = cfg.get("auxiliary") or {}
    if not isinstance(aux, dict):
        return False
    vision = aux.get("vision") or {}
    if not isinstance(vision, dict):
        return False

    provider = str(vision.get("provider") or "").strip().lower()
    model = str(vision.get("model") or "").strip()
    base_url = str(vision.get("base_url") or "").strip()

    # "auto" / "" / blank = not explicit
    if provider in ("", "auto") and not model and not base_url:
        return False
    return True


def _lookup_supports_vision(provider: str, model: str) -> Optional[bool]:
    """Return True/False if we can resolve caps, None if unknown."""
    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("image_routing: caps lookup failed for %s:%s — %s", provider, model, exc)
        return None
    if caps is None:
        return None
    return bool(caps.supports_vision)


def decide_image_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
) -> str:
    """Return ``"native"`` or ``"text"`` for the given turn.

    Args:
      provider: active inference provider ID (e.g. ``"anthropic"``, ``"openrouter"``).
      model:    active model slug as it would be sent to the provider.
      cfg:      loaded config.yaml dict, or None. When None, behaves as auto.
    """
    mode_cfg = "auto"
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent") or {}
        if isinstance(agent_cfg, dict):
            mode_cfg = _coerce_mode(agent_cfg.get("image_input_mode"))

    if mode_cfg == "native":
        return "native"
    if mode_cfg == "text":
        return "text"

    # auto
    if _explicit_aux_vision_override(cfg):
        return "text"

    supports = _lookup_supports_vision(provider, model)
    if supports is True:
        return "native"
    return "text"


# Hard ceiling for a single native image payload.  Matches vision_tools.py
# _MAX_BASE64_BYTES — the most restrictive major provider (Gemini inline
# data limit).  Oversized images are auto-resized via Pillow when available;
# if resize fails or overshoots, the image is skipped and the caller falls
# back to the text pipeline for that image.
_MAX_IMAGE_BASE64_BYTES = 20 * 1024 * 1024  # 20 MB
# Auto-resize target on first-try oversize.  5 MB aligns with Anthropic's
# per-image recommendation and comfortably fits all provider inline limits.
_RESIZE_TARGET_BYTES = 5 * 1024 * 1024


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("image/"):
        return mime
    # mimetypes on some Linux distros mis-maps .jpg; default to jpeg when
    # the suffix looks imagey.
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")


def _file_to_data_url(path: Path) -> Optional[str]:
    """Encode a local image as a base64 data URL, auto-resizing if oversized.

    Large images are downscaled via Pillow on a best-effort basis so that a
    screenshot dragged in from a 5K monitor doesn't blow context or 413 the
    provider. When Pillow isn't available and the raw file exceeds
    ``_MAX_IMAGE_BASE64_BYTES``, returns None — the caller drops this image
    from the native content parts (it gets reported back as ``skipped``).
    """
    try:
        file_size = path.stat().st_size
    except Exception as exc:
        logger.warning("image_routing: failed to stat %s — %s", path, exc)
        return None

    # Base64 expands by ~4/3.  Fast-path small images.
    estimated_b64 = (file_size * 4) // 3 + 100
    if estimated_b64 <= _MAX_IMAGE_BASE64_BYTES:
        try:
            raw = path.read_bytes()
        except Exception as exc:
            logger.warning("image_routing: failed to read %s — %s", path, exc)
            return None
        mime = _guess_mime(path)
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # Oversized — delegate to vision_tools' battle-tested resizer.
    logger.info(
        "image_routing: %s is %.1f MB, auto-resizing for native attachment",
        path.name, file_size / (1024 * 1024),
    )
    try:
        from tools.vision_tools import _resize_image_for_vision
        resized = _resize_image_for_vision(
            path,
            mime_type=_guess_mime(path),
            max_base64_bytes=_RESIZE_TARGET_BYTES,
        )
        if resized and len(resized) <= _MAX_IMAGE_BASE64_BYTES:
            return resized
        logger.warning(
            "image_routing: resize of %s did not fit under %.1f MB — "
            "dropping from native content parts",
            path.name, _MAX_IMAGE_BASE64_BYTES / (1024 * 1024),
        )
        return None
    except Exception as exc:
        logger.warning(
            "image_routing: auto-resize of %s failed (%s) — dropping", path.name, exc
        )
        return None


def build_native_content_parts(
    user_text: str,
    image_paths: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list for a user turn.

    Shape:
      [{"type": "text", "text": "..."},
       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
       ...]

    Returns (content_parts, skipped_paths). Skipped paths are files that could
    not be read (missing, permission denied, etc.); the caller can decide to
    surface a warning or fall back to text enrichment for those.
    """
    parts: List[Dict[str, Any]] = []
    skipped: List[str] = []

    text = (user_text or "").strip()
    if text:
        parts.append({"type": "text", "text": text})

    for raw_path in image_paths:
        p = Path(raw_path)
        if not p.exists() or not p.is_file():
            skipped.append(str(raw_path))
            continue
        data_url = _file_to_data_url(p)
        if not data_url:
            skipped.append(str(raw_path))
            continue
        parts.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })

    # If the text was empty, add a neutral prompt so the turn isn't just images.
    if not text and any(p.get("type") == "image_url" for p in parts):
        parts.insert(0, {"type": "text", "text": "What do you see in this image?"})

    return parts, skipped


__all__ = [
    "decide_image_input_mode",
    "build_native_content_parts",
]
