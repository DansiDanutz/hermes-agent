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


# Empirically-verified per-provider base64 ceilings.  Measured April 2026
# by sending progressively-larger PNGs through each provider's native vision
# path and observing where rejections begin:
#
#   anthropic                    → hard 5 MB (documented + HTTP 400 above)
#   openai (codex_responses)     → accepts 49 MB+ (no observed ceiling)
#   openrouter → openai/*        → accepts 49 MB+ (no observed ceiling)
#   gemini (google-genai)        → documented 100 MB inline; untested here
#   bedrock (anthropic models)   → inherits Anthropic's 5 MB
#
# When the active provider isn't in this table we do NOT impose a ceiling —
# the provider's own 400/413 is clearer than us guessing wrong.  The
# consequence of hitting a surprise ceiling is one failed turn with a
# provider-specific error message, which is recoverable; the consequence
# of us capping too aggressively is silent, permanent quality loss on
# creative workflows.
_PROVIDER_BASE64_CEILING: Dict[str, int] = {
    "anthropic": 5 * 1024 * 1024,
    "bedrock": 5 * 1024 * 1024,  # same adapter, same image source shape
}

# Target size when we do auto-resize.  Chosen to slide under Anthropic's
# 5 MB ceiling with room for headers; other providers see this as a
# no-op because their effective ceilings are much higher.
_RESIZE_TARGET_BYTES = 4 * 1024 * 1024


def _ceiling_for_provider(provider: str) -> Optional[int]:
    """Return the empirically-verified base64 ceiling for a provider, or None.

    None means "no known ceiling — let the provider police its own limit".
    """
    if not provider:
        return None
    return _PROVIDER_BASE64_CEILING.get(provider.strip().lower())


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


def _file_to_data_url(path: Path, ceiling: Optional[int] = None) -> Optional[str]:
    """Encode a local image as a base64 data URL.

    When ``ceiling`` is None (provider has no known limit), the file is
    encoded as-is — we trust the provider to return its own error if it
    disagrees.

    When ``ceiling`` is set (e.g. Anthropic's 5 MB), oversized images are
    auto-resized via Pillow to ``_RESIZE_TARGET_BYTES`` before encoding.
    If Pillow is missing or the resized output still overshoots, returns
    None so the caller can report the path in ``skipped`` and let the
    text-enrichment fallback handle it.
    """
    try:
        file_size = path.stat().st_size
    except Exception as exc:
        logger.warning("image_routing: failed to stat %s — %s", path, exc)
        return None

    # Base64 expands by ~4/3.  If we have no ceiling, or the image
    # comfortably fits, encode directly.
    estimated_b64 = (file_size * 4) // 3 + 100
    if ceiling is None or estimated_b64 <= ceiling:
        try:
            raw = path.read_bytes()
        except Exception as exc:
            logger.warning("image_routing: failed to read %s — %s", path, exc)
            return None
        mime = _guess_mime(path)
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # Oversized for this specific provider — try to resize down.
    logger.info(
        "image_routing: %s is %.1f MB (provider ceiling %.1f MB), auto-resizing",
        path.name,
        file_size / (1024 * 1024),
        ceiling / (1024 * 1024),
    )
    try:
        from tools.vision_tools import _resize_image_for_vision
        resized = _resize_image_for_vision(
            path,
            mime_type=_guess_mime(path),
            max_base64_bytes=min(_RESIZE_TARGET_BYTES, ceiling),
        )
        if resized and len(resized) <= ceiling:
            return resized
        logger.warning(
            "image_routing: resize of %s did not fit under %.1f MB — "
            "dropping from native content parts (provider would reject)",
            path.name,
            ceiling / (1024 * 1024),
        )
        return None
    except Exception as exc:
        logger.warning(
            "image_routing: auto-resize of %s failed (%s) — dropping",
            path.name,
            exc,
        )
        return None


def build_native_content_parts(
    user_text: str,
    image_paths: List[str],
    *,
    provider: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list for a user turn.

    Shape:
      [{"type": "text", "text": "..."},
       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
       ...]

    When ``provider`` is supplied and has a known ceiling (see
    ``_PROVIDER_BASE64_CEILING``), oversized images are auto-resized or
    dropped.  When provider is None or unknown, images are attached at
    their native size and the provider returns its own error if it
    disagrees — clearer and more future-proof than us guessing wrong.

    Returns (content_parts, skipped_paths). Skipped paths are files that
    couldn't be read, or that exceed the provider's ceiling even after
    resize. The caller can surface a warning or fall back to text
    enrichment for those.
    """
    parts: List[Dict[str, Any]] = []
    skipped: List[str] = []
    ceiling = _ceiling_for_provider(provider or "")

    text = (user_text or "").strip()
    if text:
        parts.append({"type": "text", "text": text})

    for raw_path in image_paths:
        p = Path(raw_path)
        if not p.exists() or not p.is_file():
            skipped.append(str(raw_path))
            continue
        data_url = _file_to_data_url(p, ceiling=ceiling)
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
