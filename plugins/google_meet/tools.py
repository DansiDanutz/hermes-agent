"""Agent-facing tools for the google_meet plugin.

Tools:
  meet_join        — join a Google Meet URL (spawns Playwright bot)
  meet_status      — report bot liveness + transcript progress
  meet_transcript  — read the current transcript (optional last-N)
  meet_leave       — signal the bot to leave cleanly
  meet_say         — v1 stub. v2 will speak through realtime audio bridge.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from plugins.google_meet import process_manager as pm


# ---------------------------------------------------------------------------
# Runtime gate
# ---------------------------------------------------------------------------

def check_meet_requirements() -> bool:
    """Return True when the plugin can actually run.

    Gates on:
      * Python ``playwright`` package importable
      * the plugin being on a supported platform (Linux or macOS)
    """
    import platform as _p
    if _p.system().lower() not in ("linux", "darwin"):
        return False
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

MEET_JOIN_SCHEMA: Dict[str, Any] = {
    "name": "meet_join",
    "description": (
        "Join a Google Meet call and start scraping live captions into a "
        "transcript file. Only meet.google.com URLs are accepted; no calendar "
        "scanning, no auto-dial. Spawns a headless Chromium subprocess that "
        "runs in parallel with the agent loop — returns immediately. Poll "
        "with meet_status and read captions with meet_transcript. Reminder "
        "to the agent: you should announce yourself in the meeting (there is "
        "no automatic consent announcement)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "Full https://meet.google.com/... URL. Required."
                ),
            },
            "guest_name": {
                "type": "string",
                "description": (
                    "Display name to use when joining as guest. Defaults to "
                    "'Hermes Agent'."
                ),
            },
            "duration": {
                "type": "string",
                "description": (
                    "Optional max duration before auto-leave (e.g. '30m', "
                    "'2h', '90s'). Omit to stay until meet_leave is called."
                ),
            },
            "headed": {
                "type": "boolean",
                "description": (
                    "Run Chromium headed instead of headless (debug only). "
                    "Default false."
                ),
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
}

MEET_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "meet_status",
    "description": (
        "Report the current Meet session state — whether the bot is alive, "
        "has joined, is sitting in the lobby, number of transcript lines "
        "captured, and last-caption timestamp."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

MEET_TRANSCRIPT_SCHEMA: Dict[str, Any] = {
    "name": "meet_transcript",
    "description": (
        "Read the scraped transcript for the active Meet session. Returns "
        "full transcript unless 'last' is set, in which case returns the last "
        "N lines only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "last": {
                "type": "integer",
                "description": (
                    "Optional: return only the last N caption lines. Useful "
                    "for polling during a meeting without re-reading the "
                    "whole transcript."
                ),
                "minimum": 1,
            },
        },
        "additionalProperties": False,
    },
}

MEET_LEAVE_SCHEMA: Dict[str, Any] = {
    "name": "meet_leave",
    "description": (
        "Leave the active Meet call cleanly, stop caption scraping, and "
        "finalize the transcript file. Safe to call when no meeting is "
        "active — returns ok=false with a reason."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

MEET_SAY_SCHEMA: Dict[str, Any] = {
    "name": "meet_say",
    "description": (
        "Speak text into the active Meet call. v1 STUB — not implemented "
        "yet. v2 will bridge through OpenAI Realtime / Gemini Live + a "
        "virtual audio device (BlackHole on macOS, PulseAudio null-sink on "
        "Linux). Today this tool returns a not-implemented error so the "
        "agent can plan around it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to speak."},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def handle_meet_join(args: Dict[str, Any], **_kw) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return _json({"success": False, "error": "url is required"})
    if not check_meet_requirements():
        return _json({
            "success": False,
            "error": (
                "google_meet plugin prerequisites missing — install with "
                "`pip install playwright && python -m playwright install "
                "chromium`. Plugin is supported on Linux and macOS only."
            ),
        })
    res = pm.start(
        url=url,
        headed=bool(args.get("headed", False)),
        guest_name=str(args.get("guest_name") or "Hermes Agent"),
        duration=str(args.get("duration")) if args.get("duration") else None,
    )
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_status(_args: Dict[str, Any], **_kw) -> str:
    res = pm.status()
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_transcript(args: Dict[str, Any], **_kw) -> str:
    last = args.get("last")
    try:
        last_i = int(last) if last is not None else None
        if last_i is not None and last_i < 1:
            last_i = None
    except (TypeError, ValueError):
        last_i = None
    res = pm.transcript(last=last_i)
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_leave(_args: Dict[str, Any], **_kw) -> str:
    res = pm.stop(reason="agent called meet_leave")
    return _json({"success": bool(res.get("ok")), **res})


def handle_meet_say(args: Dict[str, Any], **_kw) -> str:
    text = (args.get("text") or "").strip()
    return _json({
        "success": False,
        "error": (
            "meet_say is a v1 stub. Realtime duplex audio (agent speaks in "
            "the meeting) is planned for v2 via OpenAI Realtime / Gemini "
            "Live + BlackHole / PulseAudio null-sink. For now the agent "
            "can only listen (meet_transcript) and follow up outside the "
            "meeting."
        ),
        "requested_text": text,
    })
