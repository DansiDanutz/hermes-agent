"""Tests for agent/image_routing.py — the per-turn image input mode decision."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.image_routing import (
    _coerce_mode,
    _explicit_aux_vision_override,
    build_native_content_parts,
    decide_image_input_mode,
)


# ─── _coerce_mode ────────────────────────────────────────────────────────────


class TestCoerceMode:
    def test_valid_modes_pass_through(self):
        assert _coerce_mode("auto") == "auto"
        assert _coerce_mode("native") == "native"
        assert _coerce_mode("text") == "text"

    def test_case_insensitive(self):
        assert _coerce_mode("NATIVE") == "native"
        assert _coerce_mode("Auto") == "auto"

    def test_invalid_falls_back_to_auto(self):
        assert _coerce_mode("nonsense") == "auto"
        assert _coerce_mode("") == "auto"
        assert _coerce_mode(None) == "auto"
        assert _coerce_mode(42) == "auto"

    def test_strips_whitespace(self):
        assert _coerce_mode("  native  ") == "native"


# ─── _explicit_aux_vision_override ───────────────────────────────────────────


class TestExplicitAuxVisionOverride:
    def test_none_config(self):
        assert _explicit_aux_vision_override(None) is False

    def test_empty_config(self):
        assert _explicit_aux_vision_override({}) is False

    def test_default_auto_is_not_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto", "model": "", "base_url": ""}}}
        assert _explicit_aux_vision_override(cfg) is False

    def test_provider_set_is_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": ""}}}
        assert _explicit_aux_vision_override(cfg) is True

    def test_model_set_is_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto", "model": "google/gemini-2.5-flash"}}}
        assert _explicit_aux_vision_override(cfg) is True

    def test_base_url_set_is_explicit(self):
        cfg = {"auxiliary": {"vision": {"provider": "auto", "base_url": "http://localhost:11434"}}}
        assert _explicit_aux_vision_override(cfg) is True


# ─── decide_image_input_mode ─────────────────────────────────────────────────


class TestDecideImageInputMode:
    def test_explicit_native_overrides_everything(self):
        cfg = {"agent": {"image_input_mode": "native"}}
        # Non-vision model, aux-vision explicitly configured: native still wins.
        cfg["auxiliary"] = {"vision": {"provider": "openrouter", "model": "foo"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=False):
            assert decide_image_input_mode("openrouter", "some-non-vision-model", cfg) == "native"

    def test_explicit_text_overrides_everything(self):
        cfg = {"agent": {"image_input_mode": "text"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", cfg) == "text"

    def test_auto_with_vision_capable_model(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", {}) == "native"

    def test_auto_with_non_vision_model(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=False):
            assert decide_image_input_mode("openrouter", "qwen/qwen3-235b", {}) == "text"

    def test_auto_with_unknown_model(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=None):
            assert decide_image_input_mode("openrouter", "brand-new-slug", {}) == "text"

    def test_auto_respects_aux_vision_override_even_for_vision_model(self):
        """If the user configured a dedicated vision backend, don't bypass it."""
        cfg = {"auxiliary": {"vision": {"provider": "openrouter", "model": "google/gemini-2.5-flash"}}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", cfg) == "text"

    def test_none_config_is_auto(self):
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", None) == "native"

    def test_invalid_mode_coerces_to_auto(self):
        cfg = {"agent": {"image_input_mode": "weird-value"}}
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert decide_image_input_mode("anthropic", "claude-sonnet-4", cfg) == "native"


# ─── build_native_content_parts ──────────────────────────────────────────────


def _png_bytes() -> bytes:
    """Return a tiny valid 1x1 transparent PNG."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
    )


class TestBuildNativeContentParts:
    def test_text_then_image(self, tmp_path: Path):
        img = tmp_path / "cat.png"
        img.write_bytes(_png_bytes())
        parts, skipped = build_native_content_parts("hello", [str(img)])
        assert skipped == []
        assert len(parts) == 2
        assert parts[0] == {"type": "text", "text": "hello"}
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_empty_text_inserts_default_prompt(self, tmp_path: Path):
        img = tmp_path / "cat.jpg"
        img.write_bytes(_png_bytes())
        parts, skipped = build_native_content_parts("", [str(img)])
        assert skipped == []
        # Even with empty user text, we insert a neutral prompt so the turn
        # isn't just pixels.
        assert parts[0]["type"] == "text"
        assert parts[0]["text"] == "What do you see in this image?"
        assert parts[1]["type"] == "image_url"

    def test_missing_file_is_skipped(self, tmp_path: Path):
        parts, skipped = build_native_content_parts("hi", [str(tmp_path / "missing.png")])
        assert skipped == [str(tmp_path / "missing.png")]
        # Only text remains.
        assert parts == [{"type": "text", "text": "hi"}]

    def test_multiple_images(self, tmp_path: Path):
        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        img1.write_bytes(_png_bytes())
        img2.write_bytes(_png_bytes())
        parts, skipped = build_native_content_parts("compare these", [str(img1), str(img2)])
        assert skipped == []
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 2

    def test_mime_inference_jpg(self, tmp_path: Path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(_png_bytes())  # bytes are PNG but extension is jpg
        parts, _ = build_native_content_parts("x", [str(img)])
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_mime_inference_webp(self, tmp_path: Path):
        img = tmp_path / "pic.webp"
        img.write_bytes(_png_bytes())
        parts, _ = build_native_content_parts("", [str(img)])
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/webp;base64,")


# ─── Oversize handling ───────────────────────────────────────────────────────


class TestOversizeHandling:
    """Provider-aware oversize handling.

    Anthropic has a hard 5 MB base64 ceiling (empirically verified April
    2026 — returns 400 above that). OpenAI / Codex / OpenRouter accept
    49 MB+ without issue. Gemini documents 100 MB. When provider is
    unknown or has no known ceiling, we pass images through at native
    size and let the provider return its own error.
    """

    def _fake_stat_factory(self, real_stat, size_bytes):
        class _FakeStat:
            st_size = size_bytes
            st_mode = real_stat.st_mode
            st_uid = real_stat.st_uid
            st_gid = real_stat.st_gid
            st_mtime = real_stat.st_mtime
            st_ctime = real_stat.st_ctime
            st_atime = real_stat.st_atime
            st_nlink = real_stat.st_nlink
            st_ino = real_stat.st_ino
            st_dev = real_stat.st_dev
        return _FakeStat

    def test_no_ceiling_passes_through_large_image(self, tmp_path: Path, monkeypatch):
        """OpenAI / unknown provider: 30 MB image attached as-is, no resize."""
        from agent import image_routing as _ir

        img = tmp_path / "huge.png"
        img.write_bytes(_png_bytes())
        real_stat = img.stat()
        fake = self._fake_stat_factory(real_stat, 30 * 1024 * 1024)
        monkeypatch.setattr(Path, "stat", lambda self: fake())

        resize_called = {"count": 0}

        def _fake_resize(*a, **kw):
            resize_called["count"] += 1
            return "should-not-be-called"

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _fake_resize,
            raising=False,
        )

        # openai provider has no ceiling → pass through at native size.
        url = _ir._file_to_data_url(img, ceiling=None)
        assert resize_called["count"] == 0
        assert url is not None and url.startswith("data:image/png;base64,")

    def test_anthropic_ceiling_triggers_resize(self, tmp_path: Path, monkeypatch):
        """Anthropic: 10 MB image triggers auto-resize to fit under 5 MB."""
        from agent import image_routing as _ir

        img = tmp_path / "big.png"
        img.write_bytes(_png_bytes())
        real_stat = img.stat()
        fake = self._fake_stat_factory(real_stat, 10 * 1024 * 1024)
        monkeypatch.setattr(Path, "stat", lambda self: fake())

        resize_called = {"count": 0, "target": None}

        def _fake_resize(path, mime_type=None, max_base64_bytes=None):
            resize_called["count"] += 1
            resize_called["target"] = max_base64_bytes
            return "data:image/jpeg;base64,ZmFrZXJlc2l6ZWQ="

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _fake_resize,
            raising=False,
        )

        url = _ir._file_to_data_url(img, ceiling=5 * 1024 * 1024)
        assert resize_called["count"] == 1
        # Resize target should be capped at the ceiling (5 MB), not exceed it.
        assert resize_called["target"] <= 5 * 1024 * 1024
        assert url == "data:image/jpeg;base64,ZmFrZXJlc2l6ZWQ="

    def test_anthropic_ceiling_when_resize_fails(self, tmp_path: Path, monkeypatch):
        """If Pillow missing or resize overshoots, return None so caller skips."""
        from agent import image_routing as _ir

        img = tmp_path / "big.png"
        img.write_bytes(_png_bytes())
        real_stat = img.stat()
        fake = self._fake_stat_factory(real_stat, 10 * 1024 * 1024)
        monkeypatch.setattr(Path, "stat", lambda self: fake())

        def _boom(*a, **kw):
            raise RuntimeError("no Pillow")

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _boom,
            raising=False,
        )

        assert _ir._file_to_data_url(img, ceiling=5 * 1024 * 1024) is None

    def test_build_native_parts_wires_provider_ceiling(self, tmp_path: Path, monkeypatch):
        """Calling with provider='anthropic' applies the 5 MB ceiling; provider='openai' does not."""
        from agent import image_routing as _ir

        img = tmp_path / "big.png"
        img.write_bytes(_png_bytes())
        real_stat = img.stat()
        fake = self._fake_stat_factory(real_stat, 10 * 1024 * 1024)
        monkeypatch.setattr(Path, "stat", lambda self: fake())

        resize_hits = {"count": 0}

        def _fake_resize(path, mime_type=None, max_base64_bytes=None):
            resize_hits["count"] += 1
            return "data:image/jpeg;base64,UkVTSVpFRA=="

        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            _fake_resize,
            raising=False,
        )

        # anthropic → resize fires
        parts_anth, skipped_anth = _ir.build_native_content_parts(
            "hi", [str(img)], provider="anthropic"
        )
        assert resize_hits["count"] == 1
        assert skipped_anth == []
        assert any(p.get("type") == "image_url" for p in parts_anth)

        # openai → no resize, image passes through at native size
        parts_oai, skipped_oai = _ir.build_native_content_parts(
            "hi", [str(img)], provider="openai-codex"
        )
        assert resize_hits["count"] == 1  # unchanged; openai-codex has no ceiling
        assert skipped_oai == []
        # The attached URL should be the raw-encoded original (big b64),
        # not the "RESIZED" fake.
        image_part = next(p for p in parts_oai if p.get("type") == "image_url")
        assert "UkVTSVpFRA" not in image_part["image_url"]["url"]

    def test_build_native_parts_unknown_provider_no_ceiling(self, tmp_path: Path, monkeypatch):
        """Provider not in the table → no ceiling, image attached as-is."""
        from agent import image_routing as _ir

        img = tmp_path / "big.png"
        img.write_bytes(_png_bytes())
        real_stat = img.stat()
        fake = self._fake_stat_factory(real_stat, 50 * 1024 * 1024)
        monkeypatch.setattr(Path, "stat", lambda self: fake())

        resize_hits = {"count": 0}
        monkeypatch.setattr(
            "tools.vision_tools._resize_image_for_vision",
            lambda *a, **kw: resize_hits.__setitem__("count", resize_hits["count"] + 1) or "resized",
            raising=False,
        )

        parts, skipped = _ir.build_native_content_parts(
            "x", [str(img)], provider="some-future-provider"
        )
        assert resize_hits["count"] == 0
        assert skipped == []
        assert any(p.get("type") == "image_url" for p in parts)
